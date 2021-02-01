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
from macop.solutions.discrete import BinarySolution

from .LSSurrogate import LocalSearchSurrogate
from .utils.SurrogateAnalysis import SurrogateAnalysis

from sklearn.linear_model import (LinearRegression, Lasso, Lars, LassoLars,
                                    LassoCV, ElasticNet)

from wsao.sao.problems.nd3dproblem import ND3DProblem
from wsao.sao.surrogates.walsh import WalshSurrogate
from wsao.sao.algos.fitter import FitterAlgo
from wsao.sao.utils.analysis import SamplerAnalysis, FitterAnalysis, OptimizerAnalysis

class ILSMultiSpecificSurrogate(Algorithm):
    """Iterated Local Search used to avoid local optima and increave EvE (Exploration vs Exploitation) compromise using multiple-surrogate where each sub-surrogate learn from specific dataset


    Attributes:
        initalizer: {function} -- basic function strategy to initialize solution
        evaluator: {function} -- basic function in order to obtained fitness (mono or multiple objectives)
        sub_evaluator: {function} -- sub evaluator function in order to obtained fitness for sub-model
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
        solutions_folder: {str} -- Path where real evaluated solutions on subset are saved
        callbacks: {[Callback]} -- list of Callback class implementation to do some instructions every number of evaluations and `load` when initializing algorithm
    """
    def __init__(self,
                 initalizer,
                 evaluator,
                 sub_evaluator,
                 operators,
                 policy,
                 validator,
                 surrogates_file_path,
                 output_log_surrogates,
                 start_train_surrogates,
                 ls_train_surrogates,
                 k_division,
                 solutions_folder,
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
        self._sub_evaluator = sub_evaluator

        self._surrogates_file_path = surrogates_file_path
        self._start_train_surrogates = start_train_surrogates
        self._output_log_surrogates = output_log_surrogates

        self._surrogate_evaluator = None
        self._surrogate_analyser = None

        self._ls_train_surrogates = ls_train_surrogates

        self._k_division = k_division
        self._k_dynamic = k_dynamic
        self._k_random = k_random
        self._k_indices = None
        self._surrogates = None
        self._population = None

        self._generate_only = generate_only
        self._solutions_folder = solutions_folder
        

    def init_solutions_files(self):
        self._solutions_files = []

        if not os.path.exists(self._solutions_folder):
            os.makedirs(self._solutions_folder)

        # for each sub surrogate, associate its own surrogate file
        for i in range(len(self._k_indices)):
            index_str = str(i)

            while len(index_str) < 3:
                index_str = "0" + index_str

            solutions_path = os.path.join(self._solutions_folder, f'surrogate_data_{index_str}')

            # initialize solutions file if not exist
            if not os.path.exists(solutions_path):
                with open(solutions_path, 'w') as f:
                    f.write('x;y\n')

            self._solutions_files.append(solutions_path)


    def define_sub_evaluators(self): 
        self._sub_evaluators = []

        for i in range(len(self._k_indices)):

            # need to pass as default argument indices
            current_evaluator = lambda s, number=i, indices=self._k_indices[i]: self._sub_evaluator(s, number, indices)
            self._sub_evaluators.append(current_evaluator)


    def init_population(self):

        self._population = []

        # initialize the population
        for i in range(len(self._k_indices)):
            
            current_solution = self.pop_initializer(i)

            # compute fitness using sub-problem evaluator
            fitness_score = self._sub_evaluators[i](current_solution)
            current_solution._score = fitness_score
            
            self._population.append(current_solution)


    def pop_initializer(self, index):
        problem_size = len(self._k_indices[index])
        return BinarySolution([], problem_size).random(self._validator)


    def init_k_split_indices(self):
        """Initialize k_indices for the new training of surrogate

        Returns:
            k_indices: [description]
        """
        a = list(range(self._bestSolution._size))
        n_elements = int(math.ceil(self._bestSolution._size / self._k_division)) # use of ceil to avoid loss of data

        # TODO : (check) if random is possible or not
        # if self._k_random:
        #     random.shuffle(a) # random subset

        splitted_indices = [a[x:x+n_elements] for x in range(0, len(a), n_elements)]

        self._k_division = len(splitted_indices) # update size of k if necessary
        self._k_indices = splitted_indices


    def train_surrogate(self, index, indices):
        
        # 1. Data sets preparation (train and test) use now of specific dataset for surrogate
        
        # dynamic number of samples based on dataset real evaluations
        nsamples = None
        with open(self._solutions_files[index], 'r') as f:
            nsamples = len(f.readlines()) - 1 # avoid header

        training_samples = int(0.7 * nsamples) # 70% used for learning part at each iteration
        
        df = pd.read_csv(self._solutions_files[index], sep=';')
        # learning set and test set
        current_learn = df.sample(training_samples)
        current_test = df.drop(current_learn.index)

        problem = ND3DProblem(size=len(indices)) # problem size based on best solution size (need to improve...)
        model = Lasso(alpha=1e-5)
        surrogate = WalshSurrogate(order=2, size=problem.size, model=model)
        analysis = FitterAnalysis(logfile=os.path.join(self._output_log_surrogates, f"train_surrogate_{index}.log"), problem=problem)
        algo = FitterAlgo(problem=problem, surrogate=surrogate, analysis=analysis, seed=problem.seed)

        print(f"Start fitting again the surrogate model n°{index}, using {training_samples} of {nsamples} samples for train dataset")
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
        
        # 1. for each sub space indices, learn new surrogate
        if not os.path.exists(self._surrogates_file_path):
            os.makedirs(self._surrogates_file_path)

        num_cores = multiprocessing.cpu_count()

        if not os.path.exists(self._output_log_surrogates):
            os.makedirs(self._output_log_surrogates)

        Parallel(n_jobs=num_cores)(delayed(self.train_surrogate)(index, indices) for index, indices in enumerate(self._k_indices))


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

        return r_squared_scores

    def surrogates_mae(self):
        """Compute mae for each sub surrogate model

        Return:
            mae_scores: [{float}] -- mae scores from model
        """

        # for each indices set, get mae surrogate model and made prediction score
        num_cores = multiprocessing.cpu_count()

        mae_scores = Parallel(n_jobs=num_cores)(delayed(s_model.analysis.mae)(s_model.surrogate) for s_model in self._surrogates)


        return mae_scores

    def add_to_surrogate(self, solution, index):

        # save real evaluated solution into specific file for surrogate
        with open(self._solutions_files[index], 'a') as f:

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

        self.init_k_split_indices()

        # add norm to indentify sub problem data
        self.init_solutions_files()

        # here we each surrogate sub evaluator
        self.define_sub_evaluators()
        self.init_population()

        # enable resuming for ILS
        self.resume()

        # count number of surrogate obtained and restart using real evaluations done for each surrogate (sub-model)
        if (self._start_train_surrogates * self._k_division) > self.getGlobalEvaluation():

            # for each sub problem (surrogate)
            for i in range(self._k_division):

                nsamples = None
                with open(self._solutions_files[i], 'r') as f:
                    nsamples = len(f.readlines()) - 1 # avoid header

                if nsamples is None:
                    nsamples = 0

                # get `self.start_train_surrogate` number of real evaluations and save it into surrogate dataset file
                # using randomly generated solutions (in order to cover seearch space)
                while self._start_train_surrogates > nsamples:

                    print(f'Real solutions extraction for surrogate n°{i}: {nsamples} of {self._start_train_surrogates}')
                    
                    newSolution = self.pop_initializer(i)

                    # evaluate new solution
                    newSolution.evaluate(self._sub_evaluators[i])

                    # add it to surrogate pool
                    self.add_to_surrogate(newSolution, i)

                    nsamples += 1

                    # increase number of evaluation
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


            local_search_list = [] 

            for i in range(self._k_division):

                # use specific initializer for pop_initialiser
                # specific surrogate evaluator for this local search
                ls = LocalSearchSurrogate(lambda index=i: self.pop_initializer(index),
                            lambda s: self._surrogates[i].surrogate.predict([s._data])[0],
                            self._operators,
                            self._policy,
                            self._validator,
                            self._maximise,
                            parent=self)

                # add same callbacks
                for callback in self._callbacks:
                    ls.addCallback(callback)

                local_search_list.append(ls)

            # parallel run of each local search
            num_cores = multiprocessing.cpu_count()
            ls_solutions = Parallel(n_jobs=num_cores)(delayed(ls.run)(ls_evaluations) for ls in local_search_list)

            # create and search solution from local search
            self._numberOfEvaluations += ls_evaluations * self._k_division

            # for each sub problem, update population
            for i, sub_problem_solution in enumerate(ls_solutions):

                # if better solution than currently, replace it (solution saved in training pool, only if surrogate process is in a second process step)
                # Update : always add new solution into surrogate pool, not only if solution is better
                #if self.isBetter(newSolution) and self.start_train_surrogate < self.getGlobalEvaluation():
                if self._start_train_surrogates <= self.getGlobalEvaluation():

                    # if better solution found from local search, retrained the found solution and test again
                    # without use of surrogate
                    fitness_score = self._sub_evaluators[i](sub_problem_solution)
                    # self.increaseEvaluation() # dot not add evaluation

                    sub_problem_solution._score = fitness_score

                    # if solution is really better after real evaluation, then we replace (depending of problem nature (minimizing / maximizing))
                    if self._maximise:
                        if sub_problem_solution.fitness > self._population[i].fitness:
                            self._population[i] = sub_problem_solution
                    else:
                        if sub_problem_solution.fitness < self._population[i].fitness:
                            self._population[i] = sub_problem_solution

                    self.add_to_surrogate(sub_problem_solution, i)
            
            # main best solution update
            if self._start_train_surrogates <= self.getGlobalEvaluation():

                # need to create virtual solution from current population
                obtained_solution_data = np.array([ s._data for s in self._population ], dtype='object').flatten().tolist()

                if list(obtained_solution_data) == list(self._bestSolution._data):
                    print(f'-- No updates found from sub-model surrogates LS (best solution score: {self._bestSolution._score}')
                else:
                    print(f'-- Updates found into population from sub-model surrogates LS')
                    # init random solution 
                    current_solution = self._initializer()
                    current_solution.data = obtained_solution_data

                    fitness_score = self._main_evaluator(current_solution)

                    # new computed solution score
                    current_solution._score = fitness_score

                    # if solution is really better after real evaluation, then we replace
                    if self.isBetter(current_solution):
                        self._bestSolution = current_solution

                    print(f'-- Current main solution from population is {current_solution._score} vs. {self._bestSolution._score}')
                    self.progress()

            # main best solution update
            if self._start_train_surrogates <= self.getGlobalEvaluation():

                # need to create virtual solution from current population
                obtained_solution_data = np.array([ s._data for s in ls_solutions ], dtype='object').flatten().tolist()

                if list(obtained_solution_data) == list(self._bestSolution._data):
                    print(f'-- No updates found from sub-model surrogates LS (best solution score: {self._bestSolution._score}')
                else:
                    print(f'-- Updates found from sub-model surrogates LS')
                    # init random solution 
                    current_solution = self._initializer()
                    current_solution.data = obtained_solution_data

                    fitness_score = self._main_evaluator(current_solution)

                    # new computed solution score
                    current_solution._score = fitness_score

                    # if solution is really better after real evaluation, then we replace
                    if self.isBetter(current_solution):

                        print(f'Exploration solution obtained from LS surrogates enable improvment of main solution')
                        self._bestSolution = current_solution

                        print(f'Exploration solution obtained from LS surrogates enable improvment of main solution')
                        # also update the whole population as restarting process if main solution is better
                        for i, sub_problem_solution in enumerate(ls_solutions):

                            # already evaluated sub solution
                            self._population[i] = sub_problem_solution

                    print(f'-- Current main solution obtained from `LS solutions` is {current_solution._score} vs. {self._bestSolution._score}')
                    logging.info(f'-- Current main solution obtained from `LS solutions` is {current_solution._score} vs. {self._bestSolution._score}')
                    self.progress()
    
            print(f'State of current population for surrogates ({len(self._population)} members)')
            for i, s in enumerate(self._population):
                print(f'Population[{i}]: best solution fitness is {s.fitness}')

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
                
            logging.info(f"=> R² of surrogate is of {r_squared} | MAE is of {mae_score} -- [Retraining model after {self._n_local_search % training_surrogate_every} of {training_surrogate_every} LS]")
            print(f"=> R² of surrogate is of {r_squared} | MAE is of {mae_score} -- [Retraining model after {self._n_local_search % training_surrogate_every} of {training_surrogate_every} LS]")
            
            # check if necessary or not to train again surrogate
            if self._n_local_search % training_surrogate_every == 0 and self._start_train_surrogates <= self.getGlobalEvaluation():

                # reinitialization of k_indices for the new training
                # TODO : remove this part temporally
                # if self._k_dynamic:
                #     print(f"Reinitialization of k_indices using `k={self._k_division} `for the new training")
                #     self.init_k_split_indices()

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