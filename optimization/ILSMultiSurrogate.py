"""Iterated Local Search Algorithm implementation using multiple-surrogate (weighted sum surrogate) as fitness approximation
"""

# main imports
import os
import logging
import joblib
import time
import math
import numpy as np
import pandas as pd
import random

# parallel imports
from joblib import Parallel, delayed
import multiprocessing

# module imports
from macop.algorithms.base import Algorithm
from .LSSurrogate import LocalSearchSurrogate
from .utils.SurrogateAnalysis import SurrogateAnalysis

from sklearn.linear_model import (LinearRegression, Lasso, Lars, LassoLars,
                                    LassoCV, ElasticNet)

from wsao.sao.problems.nd3dproblem import ND3DProblem
from wsao.sao.surrogates.walsh import WalshSurrogate
from wsao.sao.algos.fitter import FitterAlgo
from wsao.sao.utils.analysis import SamplerAnalysis, FitterAnalysis, OptimizerAnalysis

class ILSMultiSurrogate(Algorithm):
    """Iterated Local Search used to avoid local optima and increave EvE (Exploration vs Exploitation) compromise using multiple-surrogate


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
        surrogates_file_path: {str} -- Surrogates model folder to load (models trained using https://gitlab.com/florianlprt/wsao)
        output_log_surrogates: {str} -- Log folder for surrogates training model
        start_train_surrogates: {int} -- number of evaluation expected before start training and use surrogate
        surrogates: [{Surrogate}] -- Surrogates model instance loaded
        ls_train_surrogates: {int} -- Specify if we need to retrain our surrogate model (every Local Search)
        k_division: {int} -- number of expected division for current features problem
        k_dynamic: {bool} -- specify if indices are changed for each time we train a new surrogate model
        k_random: {bool} -- random initialization of k_indices for each surrogate features model data
        generate_only: {bool} -- generate only a specific number of expected real solutions evaluated
        solutions_file: {str} -- Path where real evaluated solutions are saved in order to train surrogate again
        callbacks: {[Callback]} -- list of Callback class implementation to do some instructions every number of evaluations and `load` when initializing algorithm
    """
    def __init__(self,
                 initalizer,
                 evaluator,
                 operators,
                 policy,
                 validator,
                 surrogates_file_path,
                 output_log_surrogates,
                 start_train_surrogates,
                 ls_train_surrogates,
                 k_division,
                 solutions_file,
                 k_random=True,
                 k_dynamic=False,
                 generate_only=False,
                 maximise=True,
                 parent=None):

        # set real evaluator as default
        super().__init__(initalizer, evaluator, operators, policy,
                validator, maximise, parent)

        self._n_local_search = 0
        self._total_n_local_search = 0
        self._main_evaluator = evaluator

        self._surrogates_file_path = surrogates_file_path
        self._start_train_surrogates = start_train_surrogates
        self._output_log_surrogates = output_log_surrogates

        self._surrogate_evaluator = None
        self._surrogate_analyser = None

        self._ls_train_surrogates = ls_train_surrogates
        self._solutions_file = solutions_file

        self._k_division = k_division
        self._k_dynamic = k_dynamic
        self._k_random = k_random
        self._k_indices = None
        self._surrogates = None

        self._generate_only = generate_only

    def init_k_split_indices(self):
        """Initialize k_indices for the new training of surrogate

        Returns:
            k_indices: [description]
        """
        a = list(range(self._bestSolution._size))
        n_elements = int(math.ceil(self._bestSolution._size / self._k_division)) # use of ceil to avoid loss of data

        if self._k_random:
            random.shuffle(a) # random subset

        splitted_indices = [a[x:x+n_elements] for x in range(0, len(a), n_elements)]

        return splitted_indices


    def train_surrogate(self, index, learn, test, indices):

        current_learn = learn.copy()
        current_learn.x = current_learn.x.apply(lambda x: ','.join(list(map(str, np.fromstring(x, dtype=int, sep=',')[indices]))))

        current_test = test.copy()
        current_test.x = current_test.x.apply(lambda x: ','.join(list(map(str, np.fromstring(x, dtype=int, sep=',')[indices]))))

        problem = ND3DProblem(size=len(indices)) # problem size based on best solution size (need to improve...)
        model = Lasso(alpha=1e-5)
        surrogate = WalshSurrogate(order=2, size=problem.size, model=model)
        analysis = FitterAnalysis(logfile=os.path.join(self._output_log_surrogates, f"train_surrogate_{index}.log"), problem=problem)
        algo = FitterAlgo(problem=problem, surrogate=surrogate, analysis=analysis, seed=problem.seed)

        print(f"Start fitting again the surrogate model n°{index}")
        for r in range(10):
            print(f"Iteration n°{r}: for fitting surrogate n°{index}")
            algo.run_samples(learn=current_learn, test=current_test, step=10)

        # keep well ordered surrogate into file manager
        str_index = str(index)

        while len(str_index) < 6:
            str_index = "0" + str_index

        joblib.dump(algo, os.path.join(self._surrogates_file_path, f'surrogate_{str_index}'))

        return str_index
        

    def train_surrogates(self):
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


        # 1. Data sets preparation (train and test)
        
        # dynamic number of samples based on dataset real evaluations
        nsamples = None
        with open(self._solutions_file, 'r') as f:
            nsamples = len(f.readlines()) - 1 # avoid header

        training_samples = int(0.7 * nsamples) # 70% used for learning part at each iteration
        
        df = pd.read_csv(self._solutions_file, sep=';')
        # learning set and test set
        learn = df.sample(training_samples)
        test = df.drop(learn.index)

        print(f'Training all surrogate models using {training_samples} of {nsamples} samples for train dataset')
        
        # 2. for each sub space indices, learn new surrogate
        if not os.path.exists(self._surrogates_file_path):
            os.makedirs(self._surrogates_file_path)

        num_cores = multiprocessing.cpu_count()

        if not os.path.exists(self._output_log_surrogates):
            os.makedirs(self._output_log_surrogates)

        Parallel(n_jobs=num_cores)(delayed(self.train_surrogate)(index, learn, test, indices) for index, indices in enumerate(self._k_indices))


    def load_surrogates(self):
        """Load algorithm with surrogate model and create lambda evaluator function
        """

        # need to first train surrogate if not exist
        if not os.path.exists(self._surrogates_file_path):
            self.train_surrogates()

        self._surrogates = []

        surrogates_path = sorted(os.listdir(self._surrogates_file_path))

        for surrogate_p in surrogates_path:
            model_path = os.path.join(self._surrogates_file_path, surrogate_p)
            surrogate_model = joblib.load(model_path)

            self._surrogates.append(surrogate_model)

    
    def surrogate_evaluator(self, solution):
        """Compute mean of each surrogate model using targeted indices

        Args:
            solution: {Solution} -- current solution to evaluate using multi-surrogate evaluation

        Return:
            mean: {float} -- mean score of surrogate models
        """
        scores = []
        solution_data = np.array(solution._data)

        # for each indices set, get trained surrogate model and made prediction score
        for i, indices in enumerate(self._k_indices):
            current_data = solution_data[indices]
            current_score = self._surrogates[i].surrogate.predict([current_data])[0]
            scores.append(current_score)

        return sum(scores) / len(scores)
            
    def surrogates_coefficient_of_determination(self):
        """Compute r² for each sub surrogate model

        Return:
            r_squared_scores: [{float}] -- mean score of r_squred obtained from surrogate models
        """

        # for each indices set, get r^2 surrogate model and made prediction score

        num_cores = multiprocessing.cpu_count()

        r_squared_scores = Parallel(n_jobs=num_cores)(delayed(s_model.analysis.coefficient_of_determination)(s_model.surrogate) for s_model in self._surrogates)

        # for i, _ in enumerate(self._k_indices):
        #     r_squared = self._surrogates[i].analysis.coefficient_of_determination(self._surrogates[i].surrogate)
        #     r_squared_scores.append(r_squared)

        #print(r_squared_scores)

        return r_squared_scores

    def surrogates_mae(self):
        """Compute mae for each sub surrogate model

        Return:
            mae_scores: [{float}] -- mae scores from model
        """

        # for each indices set, get r^2 surrogate model and made prediction score

        num_cores = multiprocessing.cpu_count()

        mae_scores = Parallel(n_jobs=num_cores)(delayed(s_model.analysis.mae)(s_model.surrogate) for s_model in self._surrogates)

        # for i, _ in enumerate(self._k_indices):
        #     r_squared = self._surrogates[i].analysis.coefficient_of_determination(self._surrogates[i].surrogate)
        #     r_squared_scores.append(r_squared)

        #print(mae_scores)

        return mae_scores

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

        # initialize current solution
        self.initRun()

        # based on best solution found, initialize k pool indices
        if self._k_indices == None:
            self._k_indices = self.init_k_split_indices()

        # enable resuming for ILS
        self.resume()

        # count number of surrogate obtained and restart using real evaluations done
        nsamples = None
        with open(self._solutions_file, 'r') as f:
            nsamples = len(f.readlines()) - 1 # avoid header

        if self.getGlobalEvaluation() < nsamples:
            print(f'Restart using {nsamples} of {self._start_train_surrogates} real evaluations obtained')
            self._numberOfEvaluations = nsamples

        if self._start_train_surrogates > self.getGlobalEvaluation():
        
            # get `self.start_train_surrogate` number of real evaluations and save it into surrogate dataset file
            # using randomly generated solutions (in order to cover seearch space)
            while self._start_train_surrogates > self.getGlobalEvaluation():

                print(f'Real solutions extraction {self.getGlobalEvaluation()} of {self._start_train_surrogates}')
                
                newSolution = self._initializer()

                # evaluate new solution
                newSolution.evaluate(self._evaluator)

                # add it to surrogate pool
                self.add_to_surrogate(newSolution)

                self.increaseEvaluation()

        # stop this process after generating solution
        if self._generate_only:
            return self._bestSolution

        # train surrogate on real evaluated solutions file
        self.train_surrogates()
        self.load_surrogates()

        # local search algorithm implementation
        while not self.stop():

            # set current evaluator based on used or not of surrogate function
            self._evaluator = self.surrogate_evaluator if self._start_train_surrogates <= self.getGlobalEvaluation() else self._main_evaluator

            # create new local search instance
            # passing global evaluation param from ILS
            ls = LocalSearchSurrogate(self._initializer,
                         self._evaluator,
                         self._operators,
                         self._policy,
                         self._validator,
                         self._maximise,
                         parent=self)

            # add same callbacks
            for callback in self._callbacks:
                ls.addCallback(callback)

            # create and search solution from local search
            newSolution = ls.run(ls_evaluations)

            # if better solution than currently, replace it (solution saved in training pool, only if surrogate process is in a second process step)
            # Update : always add new solution into surrogate pool, not only if solution is better
            #if self.isBetter(newSolution) and self.start_train_surrogate < self.getGlobalEvaluation():
            if self._start_train_surrogates <= self.getGlobalEvaluation():

                # if better solution found from local search, retrained the found solution and test again
                # without use of surrogate
                fitness_score = self._main_evaluator(newSolution)
                # self.increaseEvaluation() # dot not add evaluation

                newSolution._score = fitness_score

                # if solution is really better after real evaluation, then we replace
                if self.isBetter(newSolution):
                    self._bestSolution = newSolution

                self.add_to_surrogate(newSolution)

                self.progress()

            # check using specific dynamic criteria based on r^2
            r_squared_scores = self.surrogates_coefficient_of_determination()
            r_squared = sum(r_squared_scores) / len(r_squared_scores)

            mae_scores = self.surrogates_mae()
            mae_score = sum(mae_scores) / len(mae_scores)

            r_squared_value = 0 if r_squared < 0 else r_squared

            training_surrogate_every = int(r_squared_value * self._ls_train_surrogates) # use of absolute value for r²

            # avoid issue when lauching every each local search
            if training_surrogate_every <= 0:
                training_surrogate_every = 1
                
            print(f"=> R² of surrogate is of {r_squared} | MAE is of {mae_score} -- [Retraining model after {self._n_local_search % training_surrogate_every} of {training_surrogate_every} LS]")

            # check if necessary or not to train again surrogate
            if self._n_local_search % training_surrogate_every == 0 and self._start_train_surrogates <= self.getGlobalEvaluation():

                # reinitialization of k_indices for the new training
                if self._k_dynamic:
                    print(f"Reinitialization of k_indices using `k={self._k_division} `for the new training")
                    self.init_k_split_indices()

                # train again surrogate on real evaluated solutions file
                start_training = time.time()
                self.train_surrogates()
                training_time = time.time() - start_training

                self._surrogate_analyser = SurrogateAnalysis(training_time, training_surrogate_every, r_squared_scores, r_squared, mae_scores, mae_score, self.getGlobalMaxEvaluation(), self._total_n_local_search)

                # reload new surrogate function
                self.load_surrogates()

                # reinitialize number of local search
                self._n_local_search = 0

            # increase number of local search done
            self._n_local_search += 1
            self._total_n_local_search += 1

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