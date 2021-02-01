"""Local Search algorithm
"""

# main imports
import logging

# module imports
from macop.algorithms.base import Algorithm


class LocalSearchSurrogate(Algorithm):
    """Local Search with surrogate used as exploitation optimization algorithm

    Attributes:
        initalizer: {function} -- basic function strategy to initialize solution
        evaluator: {function} -- basic function in order to obtained fitness (mono or multiple objectives)
        operators: {[Operator]} -- list of operator to use when launching algorithm
        policy: {Policy} -- Policy class implementation strategy to select operators
        validator: {function} -- basic function to check if solution is valid or not under some constraints
        maximise: {bool} -- specify kind of optimization problem 
        currentSolution: {Solution} -- current solution managed for current evaluation
        bestSolution: {Solution} -- best solution found so far during running algorithm
        callbacks: {[Callback]} -- list of Callback class implementation to do some instructions every number of evaluations and `load` when initializing algorithm
    """
    def run(self, evaluations):
        """
        Run the local search algorithm

        Args:
            evaluations: {int} -- number of Local search evaluations
            
        Returns:
            {Solution} -- best solution found
        """

        # by default use of mother method to initialize variables
        super().run(evaluations)

        # do not use here the best solution known (default use of initRun and current solution)
        # if self.parent:
        #     self.bestSolution = self.parent.bestSolution

        # initialize current solution
        # self.initRun()
        print("Inside LS => ", self._currentSolution)

        solutionSize = self._currentSolution._size

        # local search algorithm implementation
        while not self.stop():

            for _ in range(solutionSize):

                # update current solution using policy
                newSolution = self.update(self._currentSolution)

                # if better solution than currently, replace it
                if self.isBetter(newSolution):
                    self._bestSolution = newSolution

                # increase number of evaluations
                self.increaseEvaluation()

                self.progress()

                logging.info(f"---- Current {newSolution} - SCORE {newSolution.fitness}")

                # add to surrogate pool file if necessary (using ILS parent reference)
                # if self.parent.start_train_surrogate >= self.getGlobalEvaluation():
                #     self.parent.add_to_surrogate(newSolution)

                # stop algorithm if necessary
                if self.stop():
                    break

            # after applying local search on currentSolution, we switch into new local area using known current bestSolution
            self._currentSolution = self._bestSolution

        logging.info(f"End of {type(self).__name__}, best solution found {self._bestSolution}")

        return self._bestSolution

    def addCallback(self, callback):
        """Add new callback to algorithm specifying usefull parameters

        Args:
            callback: {Callback} -- specific Callback instance
        """
        # specify current main algorithm reference
        if self._parent is not None:
            callback.setAlgo(self._parent)
        else:
            callback.setAlgo(self)

        # set as new
        self._callbacks.append(callback)
