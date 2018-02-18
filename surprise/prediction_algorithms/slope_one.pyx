"""
the :mod:`slope_one` module includes the :class:`SlopeOne` algorithm.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

cimport numpy as np  # noqa
import numpy as np
from six.moves import range
from six import iteritems

from .algo_base import AlgoBase
from .predictions import PredictionImpossible


class SlopeOne(AlgoBase):
    """A simple yet accurate collaborative filtering algorithm.

    This is a straightforward implementation of the SlopeOne algorithm
    :cite:`lemire2007a`.

    The prediction :math:`\\hat{r}_{ui}` is set as:

    .. math::
        \hat{r}_{ui} = \\mu_u + \\frac{1}{
        |R_i(u)|}
        \\sum\\limits_{j \in R_i(u)} \\text{dev}(i, j),

    where :math:`R_i(u)` is the set of relevant items, i.e. the set of items
    :math:`j` rated by :math:`u` that also have at least one common user with
    :math:`i`. :math:`\\text{dev}_(i, j)` is defined as the average difference
    between the ratings of :math:`i` and those of :math:`j`:

    .. math::
        \\text{dev}(i, j) = \\frac{1}{
        |U_{ij}|}\\sum\\limits_{u \in U_{ij}} r_{ui} - r_{uj}
    """

    def __init__(self):

        AlgoBase.__init__(self)
        self.freq = None
        self.dev = None

    def train(self, trainset):

        AlgoBase.train(self, trainset)

        n_items = trainset.n_items

        # Number of users having rated items i and j: |U_ij|
        cdef np.ndarray[np.int_t, ndim=2] freq
        # Deviation from item i to item j: mean(r_ui - r_uj for u in U_ij)
        cdef np.ndarray[np.double_t, ndim=2] dev

        cdef int u, i, j, r_ui, r_uj

        AlgoBase.train(self, trainset)

        freq = np.zeros((n_items, n_items), np.int)
        dev = np.zeros((n_items, n_items), np.double)

        # Computation of freq and dev arrays.
        for u, u_ratings in iteritems(trainset.ur):
            for i, r_ui in u_ratings:
                for j, r_uj in u_ratings:
                    if i != j:
                        freq[i, j] += 1
                        dev[i, j] += r_ui - r_uj

        for i in range(n_items):
            dev[i, i] = 0
            for j in range(i + 1, n_items):
                dev[i, j] /= freq[i, j]
                dev[j, i] = -dev[i, j]

        self.freq = freq
        self.dev = dev

    def estimate(self, uiid, iiid):

        if not (self.trainset.knows_user(uiid) and self.trainset.knows_item(iiid)):
            raise PredictionImpossible('User and/or item is unkown.')

        # Ri: relevant items for i. This is the set of items j rated by u that
        # also have common users with i (i.e. at least one user has rated both
        # i and j).
        Ri = [j for (j, _) in self.trainset.ur[uiid] if self.freq[iiid, j] > 0]
        est = self.trainset.user_mean[uiid]
        if Ri:
            est += sum(self.dev[iiid, j] for j in Ri) / len(Ri)

        return est


class WeightedSlopeOne(AlgoBase):


    def __init__(self):

        AlgoBase.__init__(self)
        self.freq = None
        self.dev = None

    def train(self, trainset):

        AlgoBase.train(self, trainset)

        n_items = trainset.n_items

        # Number of users having rated items i and j: |U_ij|
        cdef np.ndarray[np.int_t, ndim=2] freq
        # Deviation from item i to item j: mean(r_ui - r_uj for u in U_ij)
        cdef np.ndarray[np.double_t, ndim=2] dev

        cdef int u, i, j, r_ui, r_uj

        AlgoBase.train(self, trainset)

        freq = np.zeros((n_items, n_items), np.int)
        dev = np.zeros((n_items, n_items), np.double)

        # Computation of freq and dev arrays.
        for u, u_ratings in iteritems(trainset.ur):
            for i, r_ui in u_ratings:
                for j, r_uj in u_ratings:
                    if i != j:
                        freq[i, j] += 1
                        dev[i, j] += r_ui - r_uj

        self.freq = freq
        self.dev = dev

    def estimate(self, uiid, iiid):

        if not (self.trainset.knows_user(uiid) and self.trainset.knows_item(iiid)):
            raise PredictionImpossible('User and/or item is unkown.')

        Ri = [j for (j, _) in self.trainset.ur[uiid] if self.freq[iiid, j] > 0]

        value = dict((j, rating) for j, rating in self.trainset.ur[uiid])

        est = self.trainset.user_mean[uiid]

        if Ri:
            num = sum(self.dev[iiid, j] + self.freq[iiid, j] * value[j] for j in Ri)
            denom = sum(self.freq[iiid, j] for j in Ri)
            est = num / denom

        return est

class BiPolarSlopeOne(AlgoBase):
    # applying my Soft Bound Bi-Polar Slope One algorithm instead

    def __init__(self):

        AlgoBase.__init__(self)
        self.freq = None
        self.dev = None

    def train(self, trainset):

        AlgoBase.train(self, trainset)

        cdef int n_items

        # Number of users having rated items i and j: |U_ij|
        cdef np.ndarray[np.int_t, ndim=2] freq

        # Deviation from item i to item j: mean(r_ui - r_uj for u in U_ij)
        cdef np.ndarray[np.double_t, ndim=2] dev

        cdef int u, i, j, r_ui, r_uj, minus

        cdef float u_mean, i_mean, j_mean

        AlgoBase.train(self, trainset)

        n_items = trainset.n_items

        freq = np.zeros((n_items, n_items), np.int)

        dev = np.zeros((n_items, n_items), np.double)

        # Computation of freq and dev arrays.
        for u, u_ratings in iteritems(trainset.ur):

            u_mean = self.trainset.user_mean[u]

            for i, r_ui in u_ratings:
                i_mean = self.trainset.item_mean[i]
                for j, r_uj in u_ratings:
                    j_mean = self.trainset.item_mean[j]
                    minus = r_ui - r_uj
                    if r_ui <= (u_mean + i_mean) / 2 + 1 and r_uj <= (u_mean + j_mean) / 2 + 1:
                        dev[i, j] += minus
                        freq[i, j] += 1

                    elif r_ui >= (u_mean + i_mean) / 2 - 1  and r_uj >= (u_mean + j_mean) / 2 - 1:
                        dev[i, j] += minus
                        freq[i, j] += 1

        self.freq = freq
        self.dev = dev

    def estimate(self, uiid, iiid):

        if not (self.trainset.knows_user(uiid) and self.trainset.knows_item(iiid)):
            raise PredictionImpossible('User and/or item is unkown.')

        value = dict((j, rating) for j, rating in self.trainset.ur[uiid])

        user_mean = self.trainset.user_mean[uiid]
        item_mean = self.trainset.item_mean[iiid]


        Ri = [j for (j, _) in self.trainset.ur[uiid] if self.freq[iiid, j] > 0]

        est = (user_mean + item_mean) / 2

        if Ri:
            num = sum(self.dev[iiid, j] + self.freq[iiid, j] * value[j] for j in Ri)
            denom = sum(self.freq[iiid, j] for j in Ri)
            est = num / denom

        return est
