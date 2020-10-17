"""Iterated Local Search Algorithm implementation using surrogate as fitness approximation
"""

# main imports
import os
import logging
import joblib

# module imports
from macop.algorithms.Algorithm import Algorithm
from .LSSurrogate import LocalSearchSurrogate

from sklearn.linear_model import (LinearRegression, Lasso, Lars, LassoLars,
                                    LassoCV, ElasticNet)

from wsao.sao.problems.nd3dproblem import ND3DProblem
from wsao.sao.surrogates.walsh import WalshSurrogate
from wsao.sao.algos.fitter import FitterAlgo
from wsao.sao.utils.analysis import SamplerAnalysis, FitterAnalysis, OptimizerAnalysis

class ILSSurrogate(Algorithm):
    """Iterated Local Search used to avoid local optima and increave EvE (Exploration vs Exploitation) compromise using surrogate


    Attributes:
        initalizer: {function} -- basic function strategy to initialize solution
        evaluator: {function} -- basic function in order to obtained fitness (mono or multiple objectives)
        operators: {[Operator]} -- list of operator to use when launching algorithm
        policy: {Policy} -- Policy class implementation strategy to select operators
        validator: {function} -- basic function to check if solution is valid or not under some constraints
        maximise: {bool} -- specify kind of optimization problem 
        currentSolution: {Solution} -- current solution managed for current evaluation
        bestSolution: {Solution} -- best solution found so far during running algorithm
        ls_iteration: {int} -- number of evaluation for each local search algorithm
        surrogate_file: {str} -- Surrogate model file to load (model trained using https://gitlab.com/florianlprt/wsao)
        start_train_surrogate: {int} -- number of evaluation expected before start training and use surrogate
        surrogate: {Surrogate} -- Surrogate model instance loaded
        ls_train_surrogate: {int} -- Specify if we need to retrain our surrogate model (every Local Search)
        solutions_file: {str} -- Path where real evaluated solutions are saved in order to train surrogate again
        callbacks: {[Callback]} -- list of Callback class implementation to do some instructions every number of evaluations and `load` when initializing algorithm
    """
    def __init__(self,
                 _initalizer,
                 _evaluator,
                 _operators,
                 _policy,
                 _validator,
                 _surrogate_file_path,
                 _start_train_surrogate,
                 _ls_train_surrogate,
                 _solutions_file,
                 _maximise=True,
                 _parent=None):

        # set real evaluator as default
        super().__init__(_initalizer, _evaluator, _operators, _policy,
                _validator, _maximise, _parent)

        self.n_local_search = 0

        self.surrogate_file_path = _surrogate_file_path
        self.start_train_surrogate = _start_train_surrogate

        self.surrogate_evaluator = None

        self.ls_train_surrogate = _ls_train_surrogate
        self.solutions_file = _solutions_file

    def train_surrogate(self):
        """etrain if necessary the whole surrogate fitness approximation function
        """
        # Following https://gitlab.com/florianlprt/wsao, we re-train the model
        # ---------------------------------------------------------------------------
        # cli_restart.py problem=nd3d,size=30,filename="data/statistics_extended_svdn" \
        #        model=lasso,alpha=1e-5 \
        #        surrogate=walsh,order=3 \
        #        algo=fitter,algo_restarts=10,samplefile=stats_extended.csv \
        #        sample=1000,step=10 \
        #        analysis=fitter,logfile=out_fit.csv

        problem = ND3DProblem(size=len(self.bestSolution.data)) # problem size based on best solution size (need to improve...)
        model = Lasso(alpha=1e-5)
        surrogate = WalshSurrogate(order=3, size=problem.size, model=model)
        analysis = FitterAnalysis(logfile="train_surrogate.log", problem=problem)

        algo = FitterAlgo(problem=problem, surrogate=surrogate, analysis=analysis, seed=problem.seed)

        print("Start fitting again the surrogate model")
        for r in range(10):
            print("Iteration n°{0}: for fitting surrogate".format(r))
            algo.run(samplefile=self.solutions_file, sample=100, step=10)

        joblib.dump(algo, self.surrogate_file_path)


    def load_surrogate(self):
        """Load algorithm with surrogate model and create lambda evaluator function
        """

        # need to first train surrogate if not exist
        if not os.path.exists(self.surrogate_file_path):
            self.train_surrogate()

        self.surrogate = joblib.load(self.surrogate_file_path)

        # update evaluator function
        self.surrogate_evaluator = lambda s: self.surrogate.surrogate.predict([s.data])[0]

    def add_to_surrogate(self, solution):

        # save real evaluated solution into specific file for surrogate
        with open(self.solutions_file, 'a') as f:

            line = ""

            for index, e in enumerate(solution.data):

                line += str(e)
                
                if index < len(solution.data) - 1:
                    line += ","

            line += ";"
            line += str(solution.score)

            f.write(line + "\n")

    def run(self, _evaluations, _ls_evaluations=100):
        """
        Run the iterated local search algorithm using local search (EvE compromise)

        Args:
            _evaluations: {int} -- number of global evaluations for ILS
            _ls_evaluations: {int} -- number of Local search evaluations (default: 100)

        Returns:
            {Solution} -- best solution found
        """

        # by default use of mother method to initialize variables
        super().run(_evaluations)

        # enable resuming for ILS
        self.resume()

        if self.start_train_surrogate < self.getGlobalEvaluation():
            self.load_surrogate()

        # initialize current solution
        self.initRun()

        # local search algorithm implementation
        while not self.stop():
            
            # set current evaluator based on used or not of surrogate function
            current_evaluator = self.surrogate_evaluator if self.start_train_surrogate < self.getGlobalEvaluation() else self.evaluator

            # create new local search instance
            # passing global evaluation param from ILS
            ls = LocalSearchSurrogate(self.initializer,
                         current_evaluator,
                         self.operators,
                         self.policy,
                         self.validator,
                         self.maximise,
                         _parent=self)

            # add same callbacks
            for callback in self.callbacks:
                ls.addCallback(callback)

            # create and search solution from local search
            newSolution = ls.run(_ls_evaluations)

            # if better solution than currently, replace it (solution saved in training pool, only if surrogate process is in a second process step)
            if self.isBetter(newSolution) and self.start_train_surrogate < self.getGlobalEvaluation():

                # if better solution found from local search, retrained the found solution and test again
                # without use of surrogate
                fitness_score = self.evaluator(newSolution)
                self.increaseEvaluation()

                newSolution.score = fitness_score

                # if solution is really better after real evaluation, then we replace
                if self.isBetter(newSolution):
                    self.bestSolution = newSolution

                self.add_to_surrogate(newSolution)


            # check if necessary or not to train again surrogate
            if self.n_local_search % self.ls_train_surrogate == 0 and self.start_train_surrogate < self.getGlobalEvaluation():

                # train again surrogate on real evaluated solutions file
                self.train_surrogate()

                # reload new surrogate function
                self.load_surrogate()

            # increase number of local search done
            self.n_local_search += 1

            self.information()

        logging.info("End of %s, best solution found %s" %
                     (type(self).__name__, self.bestSolution))

        self.end()
        return self.bestSolution