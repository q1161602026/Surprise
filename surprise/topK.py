from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict


def get_top_k(predictions, k=20):
    """Return the top-K recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        k(int): The number of recommendation to output for each user. Default
            is 20.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_k = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_k[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, item_ratings in top_k.items():
        item_ratings.sort(key=lambda x: x[1], reverse=True)
        top_k[uid] = [iid for (iid, _) in item_ratings[:k]]

    return top_k
