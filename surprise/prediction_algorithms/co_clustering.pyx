"""
the :mod:`co_clustering` module includes the :class:`CoClustering` algorithm.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

cimport numpy as np  # noqa
import numpy as np

from .algo_base import AlgoBase


class CoClustering(AlgoBase):
    """A collaborative filtering algorithm based on co-clustering.

    This is a straightforward implementation of :cite:`George:2005`.

    Basically, users and items are assigned some clusters :math:`C_u`,
    :math:`C_i`, and some co-clusters :math:`C_{ui}`.

    The prediction :math:`\\hat{r}_{ui}` is set as:

    .. math::
        \hat{r}_{ui} = \\overline{C_{ui}} + (\\mu_u - \\overline{C_u}) + (\mu_i
        - \\overline{C_i}),

    where :math:`\\overline{C_{ui}}` is the average rating of co-cluster
    :math:`C_{ui}`, :math:`\\overline{C_u}` is the average rating of
    :math:`u`'s cluster, and :math:`\\overline{C_i}` is the average rating of
    :math:`i`'s cluster. If the user is unknown, the prediction is
    :math:`\hat{r}_{ui} = \\mu_i`. If the item is unknown, the prediction is
    :math:`\hat{r}_{ui} = \\mu_u`. If both the user and the item are unknown,
    the prediction is :math:`\hat{r}_{ui} = \\mu`.

    Clusters are assigned using a straightforward optimization method, much
    like k-means.

    Args:
       cltr_u_n(int): Number of user clusters. Default is ``3``.
       cltr_i_n(int): Number of item clusters. Default is ``3``.
       n_epochs(int): Number of iteration of the optimization loop. Default is
           ``20``.
       verbose(bool): If True, the current epoch will be printed. Default is
           ``False``.

    """

    def __init__(self, cltr_u_n=3, cltr_i_n=3, n_epochs=20, seed = 0, verbose=False):

        AlgoBase.__init__(self)

        self.cltr_u_n = cltr_u_n
        self.cltr_i_n = cltr_i_n
        self.n_epochs = n_epochs
        self.verbose=verbose


        self.cltr_u = None
        self.cltr_i = None

        self.cltr_u_avg = None
        self.cltr_i_avg = None
        self.cltr_co_avg = None

        self.seed = seed

        self.user_mean = None
        self.item_mean = None

    def train(self, trainset):

        # All this implementation was hugely inspired from MyMediaLite:
        # https://github.com/zenogantner/MyMediaLite/blob/master/src/MyMediaLite/RatingPrediction/CoClustering.cs

        AlgoBase.train(self, trainset)

        # User and item means
        cdef np.ndarray[np.double_t] user_mean
        cdef np.ndarray[np.double_t] item_mean

        user_mean = self.trainset.user_mean
        item_mean = self.trainset.item_mean

        # User and items clusters
        cdef np.ndarray[np.int_t] cltr_u
        cdef np.ndarray[np.int_t] cltr_i

        # Average rating of user clusters, item clusters and co-clusters
        cdef np.ndarray[np.double_t] cltr_u_avg
        cdef np.ndarray[np.double_t] cltr_i_avg
        cdef np.ndarray[np.double_t, ndim=2] cltr_co_avg

        cdef np.ndarray[np.double_t] errors
        cdef int u, i, r, uc, ic
        cdef double est

        # Randomly assign users and items to intial clusters
        np.random.seed(self.seed)
        cltr_u = np.random.randint(self.cltr_u_n, size=trainset.n_users)
        cltr_i = np.random.randint(self.cltr_u_n, size=trainset.n_items)

        # Optimization loop. This could be optimized a bit by checking if
        # clusters where effectively updated and early stop if they did not.
        for epoch in range(self.n_epochs):

            if self.verbose:
                print("Processing epoch {}".format(epoch))

            # Update averages of clusters
            cltr_u_avg, cltr_i_avg, cltr_co_avg = self.compute_averages(cltr_u, cltr_i)
            # set user cluster to the one that minimizes squarred error of all
            # the user's ratings.
            for u in self.trainset.all_users():
                errors = np.zeros(self.cltr_u_n, np.double)
                for uc in range(self.cltr_u_n):
                    for i, r in self.trainset.ur[u]:
                        ic = cltr_i[i]
                        est = (cltr_co_avg[uc, ic] +
                               user_mean[u] - cltr_u_avg[uc] +
                               item_mean[i] - cltr_i_avg[ic])
                        errors[uc] += (r - est) ** 2
                cltr_u[u] = np.argmin(errors)

            # set item cluster to the one that minimizes squarred error over
            # all the item's ratings.
            for i in self.trainset.all_items():
                errors = np.zeros(self.cltr_i_n, np.double)
                for ic in range(self.cltr_i_n):
                    for u, r in self.trainset.ir[i]:
                        uc = cltr_u[u]
                        est = (cltr_co_avg[uc, ic] +
                               user_mean[u] - cltr_u_avg[uc] +
                               item_mean[i] - cltr_i_avg[ic])
                        errors[ic] += (r - est) ** 2
                cltr_i[i] = np.argmin(errors)

        # Compute averages one last time as clusters may have change
        cltr_u_avg, cltr_i_avg, cltr_co_avg = self.compute_averages(cltr_u, cltr_i)
        # Set cdefed arrays as attributes as they are needed for prediction
        self.cltr_u = cltr_u
        self.cltr_i = cltr_i

        self.user_mean = user_mean
        self.item_mean = item_mean

        self.cltr_u_avg = cltr_u_avg
        self.cltr_i_avg = cltr_i_avg
        self.cltr_co_avg = cltr_co_avg

    def compute_averages(self, np.ndarray[np.int_t] cltr_u,
                         np.ndarray[np.int_t] cltr_i):
        """Compute cluster averages.

        Args:
            cltr_u: current user clusters
            cltr_i: current item clusters

        Returns:
            Three arrays: averages of user clusters, item clusters and
            co-clusters.
        """

        # Number of entities in user clusters, item clusters and co-clusters.
        cdef np.ndarray[np.int_t] cltr_u_count
        cdef np.ndarray[np.int_t] cltr_i_count
        cdef np.ndarray[np.int_t, ndim=2] cltr_co_count

        # Sum of ratings for entities in each cluster
        cdef np.ndarray[np.int_t] cltr_u_sum
        cdef np.ndarray[np.int_t] sum_cltr_i
        cdef np.ndarray[np.int_t, ndim=2] cltr_co_sum

        # The averages of each cluster (what will be returned)
        cdef np.ndarray[np.double_t] cltr_u_avg
        cdef np.ndarray[np.double_t] cltr_i_avg
        cdef np.ndarray[np.double_t, ndim=2] cltr_co_avg

        cdef int u, i, r, uc, ic
        cdef double global_mean = self.trainset.global_mean

        # Initialize everything to zero
        cltr_u_count = np.zeros(self.cltr_u_n, np.int)
        cltr_i_count = np.zeros(self.cltr_i_n, np.int)
        cltr_co_count = np.zeros((self.cltr_u_n, self.cltr_i_n), np.int)

        cltr_u_sum = np.zeros(self.cltr_u_n, np.int)
        cltr_i_sum = np.zeros(self.cltr_i_n, np.int)
        cltr_co_sum = np.zeros((self.cltr_u_n, self.cltr_i_n), np.int)

        cltr_u_avg = np.zeros(self.cltr_u_n, np.double)
        cltr_i_avg = np.zeros(self.cltr_i_n, np.double)
        cltr_co_avg = np.zeros((self.cltr_u_n, self.cltr_i_n), np.double)

        # Compute counts and sums for every cluster.
        for u, i, r in self.trainset.all_ratings():
            uc = cltr_u[u]
            ic = cltr_i[i]

            cltr_u_count[uc] += 1
            cltr_i_count[ic] += 1
            cltr_co_count[uc, ic] += 1

            cltr_u_sum[uc] += r
            cltr_i_sum[ic] += r
            cltr_co_sum[uc, ic] += r

        # Then set the averages for users...
        for uc in range(self.cltr_u_n):
            if cltr_u_count[uc]:
                cltr_u_avg[uc] = cltr_u_sum[uc] / cltr_u_count[uc]
            else:
                cltr_u_avg[uc] = global_mean

        # ... for items
        for ic in range(self.cltr_i_n):
            if cltr_i_count[ic]:
                cltr_i_avg[ic] = cltr_i_sum[ic] / cltr_i_count[ic]
            else:
                cltr_i_avg[ic] = global_mean

        # ... and for co-clusters
        for uc in range(self.cltr_u_n):
            for ic in range(self.cltr_i_n):
                if cltr_co_count[uc, ic]:
                    cltr_co_avg[uc, ic] = (cltr_co_sum[uc, ic] /
                                          cltr_co_count[uc, ic])
                else:
                    cltr_co_avg[uc, ic] = global_mean

        return cltr_u_avg, cltr_i_avg, cltr_co_avg

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            return self.trainset.global_mean

        if not self.trainset.knows_user(u):
            return self.cltr_i[i]

        if not self.trainset.knows_item(i):
            return self.cltr_u[u]

        # I doubt cdefing makes any difference here as cython has no clue about
        # arrays self.stuff... But maybe?
        cdef int _u = u
        cdef int _i = i
        cdef int uc = self.cltr_u[_u]
        cdef int ic = self.cltr_i[_i]
        cdef double est

        est = (self.cltr_co_avg[uc, ic] +
               self.user_mean[_u] - self.cltr_u_avg[uc] +
               self.item_mean[_i] - self.cltr_i_avg[ic])

        return est
