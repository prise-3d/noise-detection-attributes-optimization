"""Iterated Local Search Algorithm implementation using surrogate as fitness approximation
"""

# main imports
import os
import logging
import joblib
import time

# module imports
from macop.algorithms.base import Algorithm
from macop.evaluators.base import Evaluator
from macop.operators.base import KindOperator
from macop.policies.reinforcement import UCBPolicy

from macop.callbacks.policies import UCBCheckpoint

from .LSSurrogate import LocalSearchSurrogate
from .utils.SurrogateAnalysis import SurrogateAnalysisMono

from sklearn.linear_model import (LinearRegression, Lasso, Lars, LassoLars,
                                    LassoCV, ElasticNet)

from wsao.sao.problems.nd3dproblem import ND3DProblem
from wsao.sao.surrogates.walsh import WalshSurrogate
from wsao.sao.algos.fitter import FitterAlgo
from wsao.sao.utils.analysis import SamplerAnalysis, FitterAnalysis, OptimizerAnalysis


class LSSurrogateEvaluator(Evaluator):

    # use of surrogate in order to evaluate solution
    def compute(self, solution):
        return self._data['surrogate'].surrogate.predict([solution.data])[0]
        

class ILSPopSurrogate(Algorithm):
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
        population_size: {int} -- size of the population to manage
        surrogate_file: {str} -- Surrogate model file to load (model trained using https://gitlab.com/florianlprt/wsao)
        start_train_surrogate: {int} -- number of evaluation expected before start training and use surrogate
        surrogate: {Surrogate} -- Surrogate model instance loaded
        ls_train_surrogate: {int} -- Specify if we need to retrain our surrogate model (every Local Search)
        solutions_file: {str} -- Path where real evaluated solutions are saved in order to train surrogate again
        callbacks: {[Callback]} -- list of Callback class implementation to do some instructions every number of evaluations and `load` when initializing algorithm
    """
    def __init__(self,
                 initalizer,
                 evaluator,
                 operators,
                 policy,
                 validator,
                 population_size,
                 surrogate_file_path,
                 start_train_surrogate,
                 ls_train_surrogate,
                 walsh_order,
                 inter_policy_ls_file,
                 solutions_file,
                 maximise=True,
                 parent=None):

        # set real evaluator as default
        super().__init__(initalizer, evaluator, operators, policy,
                validator, maximise, parent)

        self._n_local_search = 0
        self._main_evaluator = evaluator

        self._surrogate_file_path = surrogate_file_path
        self._start_train_surrogate = start_train_surrogate

        self._surrogate_evaluator = None
        self._surrogate_analyser = None

        self._ls_train_surrogate = ls_train_surrogate
        self._solutions_file = solutions_file

        self._walsh_order = walsh_order
        self._inter_policy_ls_file = inter_policy_ls_file

        # default population values
        self.population_size = population_size
        self.population = []

        for _ in range(self.population_size):
            self.population.append(None)

    def train_surrogate(self):
        """Retrain if necessary the whole surrogate fitness approximation function
        """
        # Following https://gitlab.com/florianlprt/wsao, we re-train the model
        # ---------------------------------------------------------------------------
        # cli_restart.py problem=nd3d,size=30,filename="data/statistics_extended_svdn" \
        #        model=lasso,alpha=1e-5 \
        #        surrogate=walsh,order=3 \
        #        algo=fitter,algo_restarts=10,samplefile=stats_extended.csv \
        #        sample=1000,step=10 \
        #        analysis=fitter,logfile=out_fit.csv

        problem = ND3DProblem(size=len(self._bestSolution.data)) # problem size based on best solution size (need to improve...)
        model = Lasso(alpha=1e-5)
        surrogate = WalshSurrogate(order=self._walsh_order, size=problem.size, model=model)
        analysis = FitterAnalysis(logfile="train_surrogate.log", problem=problem)
        algo = FitterAlgo(problem=problem, surrogate=surrogate, analysis=analysis, seed=problem.seed)

        # dynamic number of samples based on dataset real evaluations
        nsamples = None
        with open(self._solutions_file, 'r') as f:
            nsamples = len(f.readlines()) - 1 # avoid header

        training_samples = int(0.7 * nsamples) # 70% used for learning part at each iteration
        
        print("Start fitting again the surrogate model")
        print(f'Using {training_samples} of {nsamples} samples for train dataset')
        for r in range(10):
            print(f"Iteration n°{r}: for fitting surrogate")
            algo.run(samplefile=self._solutions_file, sample=training_samples, step=10)

        joblib.dump(algo, self._surrogate_file_path)


    def load_surrogate(self):
        """Load algorithm with surrogate model and create lambda evaluator function
        """

        # need to first train surrogate if not exist
        if not os.path.exists(self._surrogate_file_path):
            self.train_surrogate()

        self._surrogate = joblib.load(self._surrogate_file_path)

        # update evaluator function
        self._surrogate_evaluator = LSSurrogateEvaluator(data={'surrogate': self._surrogate})

    def add_to_surrogate(self, solution):

        # save real evaluated solution into specific file for surrogate
        with open(self._solutions_file, 'a') as f:

            line = ""

            for index, e in enumerate(solution._data):

                line += str(e)
                
                if index < len(solution._data) - 1:
                    line += ","

            line += ";"
            line += str(solution._score)

            f.write(line + "\n")

    def initRun(self):

        fitness_scores = []
        print('Initialisation of @population')
        for i in range(len(self.population)):

            print(f'  - solution [{(i+1)}] of {len(self.population)}')
            if self.population[i] is None:
                solution = self.initialiser()
                solution.evaluate(self.evaluator)

                self.population[i] = solution
                self.add_to_surrogate(solution)

            self.increaseEvaluation()

            fitness_scores.append(self.population[i].fitness)

        print('Best solution @initialisation')
        self._bestSolution = self.population[fitness_scores.index(max(fitness_scores))]


    def run(self, evaluations, ls_evaluations=100):
        """
        Run the iterated local search algorithm using local search (EvE compromise)

        Args:
            evaluations: {int} -- number of global evaluations for ILS
            ls_evaluations: {int} -- number of Local search evaluations (default: 100)

        Returns:
            {Solution} -- best solution found
        """

        # by default use of mother method to initialize variables
        super().run(evaluations)

        # enable resuming for ILS
        self.resume()

        # initialize current solution
        self.initRun()

        # count number of surrogate obtained and restart using real evaluations done
        nsamples = None
        with open(self._solutions_file, 'r') as f:
            nsamples = len(f.readlines()) - 1 # avoid header

        if self.getGlobalEvaluation() < nsamples:
            print(f'Restart using {nsamples} of {self._start_train_surrogate} real evaluations obtained')
            self._numberOfEvaluations = nsamples

        if self._start_train_surrogate > self.getGlobalEvaluation():
        
            # get `self.start_train_surrogate` number of real evaluations and save it into surrogate dataset file
            # using randomly generated solutions (in order to cover seearch space)
            while self._start_train_surrogate > self.getGlobalEvaluation():
                
                newSolution = self.initialiser()

                # evaluate new solution
                newSolution.evaluate(self.evaluator)

                # add it to surrogate pool
                self.add_to_surrogate(newSolution)

                self.increaseEvaluation()

        # train surrogate on real evaluated solutions file
        self.train_surrogate()
        self.load_surrogate()

        # local search algorithm implementation
        while not self.stop():

            # set current evaluator based on used or not of surrogate function
            self.evaluator = self._surrogate_evaluator if self._start_train_surrogate <= self.getGlobalEvaluation() else self._main_evaluator

            for i in range(len(self.population)):

                # pass only Mutators operators for local search
                selected_operators = [ op for op in self._operators if op._kind == KindOperator.MUTATOR ]

                ls_policy = UCBPolicy(selected_operators, C=100, exp_rate=0.1)
                # create new local search instance
                # passing global evaluation param from ILS
                ls = LocalSearchSurrogate(self.initialiser,
                            self.evaluator,
                            selected_operators,
                            ls_policy,
                            self.validator,
                            self._maximise,
                            parent=None,
                            verbose=False)

                ls.addCallback(UCBCheckpoint(every=1, filepath=self._inter_policy_ls_file))

                # create current new solution using policy and custom algorithm init
                ls._currentSolution = self.policy.apply(self.population[i])
                ls.result = ls._currentSolution

                # add same callbacks
                #for callback in self._callbacks:
                #    ls.addCallback(callback)

                # create and search solution from local search
                newSolution = ls.run(ls_evaluations)

                # if better solution than currently, replace it (solution saved in training pool, only if surrogate process is in a second process step)
                # Update : always add new solution into surrogate pool, not only if solution is better
                #if self.isBetter(newSolution) and self.start_train_surrogate < self.getGlobalEvaluation():
                if self._start_train_surrogate <= self.getGlobalEvaluation():

                    # if better solution found from local search, retrained the found solution and test again
                    # without use of surrogate
                    fitness_score = self._main_evaluator.compute(newSolution)
                    # self.increaseEvaluation() # dot not add evaluation

                    newSolution.fitness = fitness_score

                    # if solution is really better after real evaluation, then we replace
                    if self.isBetter(newSolution):
                        self.result = newSolution

                    # update population
                    if self.population[i].fitness < newSolution.fitness:
                        self.population[i] = newSolution

                    self.add_to_surrogate(newSolution)

                    self.progress()
                    
                self.increaseEvaluation()

                print(f'=================================================================')
                print(f'Best solution found so far: {self.result.fitness}')

                # check using specific dynamic criteria based on r^2
                r_squared = self._surrogate.analysis.coefficient_of_determination(self._surrogate.surrogate)
                mae = self._surrogate.analysis.mae(self._surrogate.surrogate)
                training_surrogate_every = int(r_squared * self._ls_train_surrogate)
                print(f"=> R² of surrogate is of {r_squared}.")
                print(f"=> MAE of surrogate is of {mae}.")
                print(f'=> Retraining model every {training_surrogate_every} LS ({self._n_local_search} of {training_surrogate_every})')
                # avoid issue when lauching every each local search
                if training_surrogate_every <= 0:
                    training_surrogate_every = 1

                # check if necessary or not to train again surrogate
                if self._n_local_search % training_surrogate_every == 0 and self._start_train_surrogate <= self.getGlobalEvaluation():

                    # train again surrogate on real evaluated solutions file
                    start_training = time.time()
                    self.train_surrogate()
                    training_time = time.time() - start_training

                    self._surrogate_analyser = SurrogateAnalysisMono(training_time, training_surrogate_every, r_squared, mae, self.getGlobalMaxEvaluation(), self._n_local_search)

                    # reload new surrogate function
                    self.load_surrogate()

                # increase number of local search done
                self._n_local_search += 1

                self.information()

        logging.info(f"End of {type(self).__name__}, best solution found {self._bestSolution}")

        self.end()
        return self._bestSolution

    def addCallback(self, callback):
        """Add new callback to algorithm specifying usefull parameters

        Args:
            callback: {Callback} -- specific Callback instance
        """
        # specify current main algorithm reference
        if self.getParent() is not None:
            callback.setAlgo(self.getParent())
        else:
            callback.setAlgo(self)

        # set as new
        self._callbacks.append(callback)