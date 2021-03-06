"""
The :mod:`surprise.prediction_algorithms.algo_base` module defines the base
class :class:`AlgoBase` from which every single prediction algorithm has to
inherit.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from .. import accuracy

from .predictions import PredictionImpossible
from .predictions import Prediction
from .optimize_baselines import baseline_als
from .optimize_baselines import baseline_sgd


class AlgoBase:
    """Abstract class where is defined the basic behavior of a prediction
    algorithm.

    Keyword Args:
        baseline_options(dict, optional): If the algorithm needs to compute a
            baseline estimate, the ``baseline_options`` parameter is used to
            configure how they are computed. See
            :ref:`baseline_estimates_configuration` for usage.
    """

    def __init__(self, **kwargs):

        self.trainset = None
        self.bsl_options = kwargs.get('bsl_options', {})
        self.bu = None
        self.bi = None

    def train(self, trainset):
        """Train an algorithm on a given training set.

        This method is called by every derived class as the first basic step
        for training an algorithm. It basically just initializes some internal
        structures and set the self.trainset attribute.

        Args:
            trainset(:obj:`Trainset <surprise.dataset.Trainset>`) : A training
                set, as returned by the :meth:`folds
                <surprise.dataset.Dataset.folds>` method.
        """
        self.trainset = trainset
        # (re) Initialise baselines
        self.bu = None
        self.bi = None

    def estimate(self, uiid, iiid):
        pass

    def predict(self, urid, irid, r_ui=None, clip=True, verbose=False):
        """Compute the rating prediction for given user and item.

        The ``predict`` method converts raw ids to inner ids and then calls the
        ``estimate`` method which is defined in every derived class. If the
        prediction is impossible (for whatever reason), the prediction is set
        to the global mean of all ratings.

        Args:
            urid: (Raw) id of the user. See :ref:`this note<raw_inner_note>`.
            irid: (Raw) id of the item. See :ref:`this note<raw_inner_note>`.
            r_ui(float): The true rating :math:`r_{ui}`. Optional, default is
                ``None``.
            clip(bool): Whether to clip the estimation into the rating scale.
                For example, if :math:`\\hat{r}_{ui}` is :math:`5.5` while the
                rating scale is :math:`[1, 5]`, then :math:`\\hat{r}_{ui}` is
                set to :math:`5`. Same goes if :math:`\\hat{r}_{ui} < 1`.
                Default is ``True``.
            verbose(bool): Whether to print details of the prediction.  Default
                is False.

        Returns:
            A :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>` object
            containing:

            - The (raw) user id ``uid``.
            - The (raw) item id ``iid``.
            - The true rating ``r_ui`` (:math:`\\hat{r}_{ui}`).
            - The estimated rating (:math:`\\hat{r}_{ui}`).
            - Some additional details about the prediction that might be useful
              for later analysis.
        """

        # Convert raw ids to inner ids
        try:
            uiid = self.trainset.to_inner_uid(urid)
        except ValueError:
            uiid = 'UKN__' + str(urid)
        try:
            iiid = self.trainset.to_inner_iid(irid)
        except ValueError:
            iiid = 'UKN__' + str(irid)
        details = {}
        try:
            est = self.estimate(uiid, iiid)

            # If the details dict was also returned
            if isinstance(est, tuple):
                est, details = est

            details['was_impossible'] = False

        except PredictionImpossible as e:
            est = self.trainset.global_mean
            details['was_impossible'] = True
            details['reason'] = str(e)

        # Remap the rating into its initial rating scale (because the rating
        # scale was translated so that ratings are all >= 1)
        if est is not None:
            est -= self.trainset.offset

            # clip estimate into [lower_bound, higher_bound]
            if clip:
                lower_bound, higher_bound = self.trainset.rating_scale
                est = min(higher_bound, est)
                est = max(lower_bound, est)

        pred = Prediction(urid, irid, r_ui, est, details)

        if verbose:
            print(pred)

        return pred

    def test(self, testset, verbose=False):
        """Test the algorithm on given testset, i.e. estimate all the ratings
        in the given testset.

        Args:
            testset: A test set, as returned by the :meth:`folds()
                <surprise.dataset.Dataset.folds>` method or by the
                :meth:`build_testset()
                <surprise.dataset.Trainset.build_testset>` method.
            verbose(bool): Whether to print details for each predictions.
                Default is False.

        Returns:
            A list of :class:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>` objects
            that contains all the estimated ratings.
        """

        # The ratings are translated back to their original scale.
        predictions = [self.predict(urid,
                                    irid,
                                    r_ui_trans - self.trainset.offset,
                                    verbose=verbose)
                       for (urid, irid, r_ui_trans) in testset.build_raw_testset()]
        return predictions

    def compute_baselines(self):
        """Compute users and items baselines.

        The way baselines are computed depends on the ``bsl_options`` parameter
        passed at the creation of the algorithm (see
        :ref:`baseline_estimates_configuration`).

        This method is only relevant for algorithms using :func:`Pearson
        baseline similarty<surprise.similarities.pearson_baseline>` or the
        :class:`BaselineOnly
        <surprise.prediction_algorithms.baseline_only.BaselineOnly>` algorithm.

        Returns:
            A tuple ``(bu, bi)``, which are users and items baselines."""

        # Firt of, if this method has already been called before on the same
        # trainset, then just return. Indeed, compute_baselines may be called
        # more than one time, for example when a similarity metric (e.g.
        # pearson_baseline) uses baseline estimates.
        if self.bu is not None:
            return self.bu, self.bi

        method = dict(als=baseline_als,
                      sgd=baseline_sgd)

        method_name = self.bsl_options.get('method', 'als')

        try:
            print('Estimating biases using', method_name + '...')
            self.bu, self.bi = method[method_name](self)
            return self.bu, self.bi
        except KeyError:
            raise ValueError('Invalid method ' + method_name +
                             ' for baseline computation.' +
                             ' Available methods are als and sgd.')

    def evaluate(self, testset, measures={'rmse', 'mae'}, verbose=1):
        """Evaluate the performance of the algorithm on given data.

        Depending on the nature of the ``data`` parameter, it may or may not
        perform cross validation.

        Args:
            algo(:obj:`AlgoBase \
                <surprise.prediction_algorithms.algo_base.AlgoBase>`):
                The algorithm to evaluate.
            testset(:obj:`Dataset <surprise.dataset.Testset>`): The dataset on which
                to evaluate the algorithm.
            measures(set of string): The performance measures to compute. Allowed
                names are function names as defined in the :mod:`accuracy
                <surprise.accuracy>` module. Default is ``{'rmse', 'mae'}``.
            verbose(int): Level of verbosity. If 0, nothing is printed. If 1
                (default), accuracy measures for each folds are printed, with a
                final summary. If 2, every prediction is printed.

        Returns:
            A dictionary containing measures as keys and lists as values. Each list
            contains one entry per fold.
        """

        from ..evaluate import CaseInsensitiveDefaultDict
        performances = CaseInsensitiveDefaultDict(list)

        if verbose:
            print('Evaluating {0} of algorithm {1}.'.format(
                  ', '.join((m.upper() for m in measures)),
                  self.__class__.__name__))
            print()

        predictions = self.test(testset, verbose=(verbose == 2))

        # compute needed performance statistics
        for measure in measures:
            f = getattr(accuracy, measure.lower())
            performances[measure].append(f(predictions, verbose=verbose))

        return performances