"""
The :mod:`surprise.accuracy` module provides with tools for computing accuracy
metrics on a set of predictions.

Available accuracy metrics:

.. autosummary::
    :nosignatures:

    rmse
    mae
    fcp
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict
import numpy as np
from six import iteritems
from .topK import get_top_k


def rmse(predictions, verbose=True):
    """Compute RMSE (Root Mean Squared Error).

    .. math::
        \\text{RMSE} = \\sqrt{\\frac{1}{|\\hat{R}|} \\sum_{\\hat{r}_{ui} \in
        \\hat{R}}(r_{ui} - \\hat{r}_{ui})^2}.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Root Mean Squared Error of predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    mse = np.mean([float((true_r - est)**2)
                   for (_, _, true_r, est, _) in predictions])
    rmse_ = np.sqrt(mse)

    if verbose:
        print('RMSE: {0:1.4f}'.format(rmse_))

    return rmse_


def mae(predictions, verbose=True):
    """Compute MAE (Mean Absolute Error).

    .. math::
        \\text{MAE} = \\frac{1}{|\\hat{R}|} \\sum_{\\hat{r}_{ui} \in
        \\hat{R}}|r_{ui} - \\hat{r}_{ui}|

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Mean Absolute Error of predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    mae_ = np.mean([float(abs(true_r - est))
                    for (_, _, true_r, est, _) in predictions])

    if verbose:
        print('MAE:  {0:1.4f}'.format(mae_))

    return mae_


def fcp(predictions, verbose=True):
    """Compute FCP (Fraction of Concordant Pairs).

    Computed as described in paper `Collaborative Filtering on Ordinal User
    Feedback <http://www.ijcai.org/Proceedings/13/Papers/449.pdf>`_ by Koren
    and Sill, section 5.2.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Fraction of Concordant Pairs.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    predictions_u = defaultdict(list)
    nc_u = defaultdict(int)
    nd_u = defaultdict(int)

    for u0, _, r0, est, _ in predictions:
        predictions_u[u0].append((r0, est))

    for u0, preds in iteritems(predictions_u):
        for r0i, esti in preds:
            for r0j, estj in preds:
                if esti > estj and r0i > r0j:
                    nc_u[u0] += 1
                if esti >= estj and r0i < r0j:
                    nd_u[u0] += 1

    nc = np.mean(list(nc_u.values())) if nc_u else 0
    nd = np.mean(list(nd_u.values())) if nd_u else 0

    try:
        fcp = nc / (nc + nd)
    except ZeroDivisionError:
        raise ValueError('cannot compute fcp on this list of prediction. ' +
                         'Does every user have at least two predictions?')

    if verbose:
        print('FCP:  {0:1.4f}'.format(fcp))

    return fcp


def precision_recall_score(testset, predictions, k=20, verbose=True):

    hit = 0
    recommend_count = 0
    rated_count = len(testset)

    user_top_k = get_top_k(predictions=predictions, k=k)
    user_recommended = {}
    for ruid, riid, _ in testset:
        flag = user_recommended.get(ruid, False)
        if not flag:
            recommend_count += len(user_top_k[ruid])
            user_recommended[ruid] = True
        if riid in user_top_k[ruid]:
            hit += 1


    precision_score = hit / (1.0 * recommend_count)
    recall_score = hit / (1.0 * rated_count)
    if verbose:
        print('Precision:  {0:1.4f}'.format(precision_score))
        print('Recall:  {0:1.4f}'.format(recall_score))
    return precision_score, recall_score


def precision(testset, predictions, k=20, verbose=True):

    hit = 0
    recommend_count = 0

    ur = defaultdict(list)

    # user raw id, item raw id, translated rating, time stamp
    for uid, iid, _ in testset:
        ur[uid].append(iid)

    user_top_k = get_top_k(predictions=predictions, k=k)
    for uid, iid_list in ur.items():
        recommend_list = user_top_k[uid]
        recommend_count += len(recommend_list)
        for item in recommend_list:
            if item in iid_list:
                hit += 1

    precision_score = hit / (1.0 * recommend_count)
    if verbose:
        print('Precision:  {0:1.4f}'.format(precision_score))
    return precision_score


def recall(testset, predictions, k=20, verbose=True):

    hit = 0
    rated_count = len(testset)

    user_top_k = get_top_k(predictions=predictions, k=k)
    for ruid, riid, _ in testset:
        if riid in user_top_k[ruid]:
            hit += 1

    recall_score = hit / (1.0 * rated_count)
    if verbose:
        print('Recall:  {0:1.4f}'.format(recall_score))
    return recall_score
