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

        freq = np.zeros((trainset.n_items, trainset.n_items), np.int)
        dev = np.zeros((trainset.n_items, trainset.n_items), np.double)

        # Computation of freq and dev arrays.
        for u, u_ratings in iteritems(trainset.ur):
            for i, r_ui in u_ratings:
                for j, r_uj in u_ratings:
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

        freq = np.zeros((trainset.n_items, trainset.n_items), np.int)
        dev = np.zeros((trainset.n_items, trainset.n_items), np.double)

        # Computation of freq and dev arrays.
        for u, u_ratings in iteritems(trainset.ur):
            for i, r_ui in u_ratings:
                for j, r_uj in u_ratings:
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

        Ri = [j for (j, _) in self.trainset.ur[uiid] if self.freq[iiid, j] > 0]

        value = dict((j, rating) for j, rating in self.trainset.ur[uiid])

        est = self.trainset.user_mean[uiid]

        if Ri:
            num = sum(self.freq[iiid, j] * (self.dev[iiid, j] + value[j]) for j in Ri)
            denom = sum(self.freq[iiid, j] for j in Ri)
            est = num / denom

        return est

class BiPolarSlopeOne(AlgoBase):

    def __init__(self):

        AlgoBase.__init__(self)
        self.freq1 = None
        self.freq2 = None
        self.dev1 = None
        self.dev2 = None

    def train(self, trainset):

        AlgoBase.train(self, trainset)

        n_items = trainset.n_items

        # Number of users having rated items i and j: |U_ij|
        cdef np.ndarray[np.int_t, ndim=2] freq1
        cdef np.ndarray[np.int_t, ndim=2] freq2

        # Deviation from item i to item j: mean(r_ui - r_uj for u in U_ij)
        cdef np.ndarray[np.double_t, ndim=2] dev1
        cdef np.ndarray[np.double_t, ndim=2] dev2

        cdef int u, i, j, r_ui, r_uj

        AlgoBase.train(self, trainset)

        freq1 = np.zeros((trainset.n_items, trainset.n_items), np.int)
        freq2 = np.zeros((trainset.n_items, trainset.n_items), np.int)

        dev1 = np.zeros((trainset.n_items, trainset.n_items), np.double)
        dev2 = np.zeros((trainset.n_items, trainset.n_items), np.double)

        # Computation of freq and dev arrays.
        for u, u_ratings in iteritems(trainset.ur):

            u_mean = self.trainset.user_mean[u]

            for i, r_ui in u_ratings:
                for j, r_uj in u_ratings:
                    minus = r_ui - r_uj

                    if r_ui > u_mean and r_uj > u_mean:
                        dev1[i, j] += minus
                        freq1[i, j] += 1

                    elif r_ui < u_mean and r_uj < u_mean:
                        dev2[i, j] += minus
                        freq2[i, j] += 1


        for i in range(n_items):
            dev1[i, i] = 0
            dev2[i, i] = 0

            for j in range(i + 1, n_items):
                dev1[i, j] /= freq1[i, j]
                dev2[i, j] /= freq2[i, j]

                dev1[j, i] = -dev1[i, j]
                dev2[j, i] = -dev2[i, j]


        self.freq1 = freq1
        self.dev1 = dev1
        self.freq2 = freq2
        self.dev2 = dev2

    def estimate(self, uiid, iiid):

        if not (self.trainset.knows_user(uiid) and self.trainset.knows_item(iiid)):
            raise PredictionImpossible('User and/or item is unkown.')

        value = dict((j, rating) for j, rating in self.trainset.ur[uiid])

        user_mean = self.trainset.user_mean[uiid]

        Ri1 = [j for (j, _) in self.trainset.ur[uiid] if self.freq1[iiid, j] > 0]
        Ri2 = [j for (j, _) in self.trainset.ur[uiid] if self.freq2[iiid, j] > 0]

        est = user_mean

        if Ri1 or Ri2:
            num1 = sum(self.freq1[iiid, j] * (self.dev1[iiid, j] + value[j]) for j in Ri1)
            denom1 = sum(self.freq1[iiid, j] for j in Ri1)
            num2 = sum(self.freq2[iiid, j] * (self.dev2[iiid, j] + value[j]) for j in Ri2)
            denom2 = sum(self.freq2[iiid, j] for j in Ri2)
            est = (num1 + num2) / (denom1 + denom2)

        return est
