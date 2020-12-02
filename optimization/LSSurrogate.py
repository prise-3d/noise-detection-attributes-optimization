"""Local Search algorithm
"""

# main imports
import logging

# module imports
from macop.algorithms.Algorithm import Algorithm


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
    def run(self, _evaluations):
        """
        Run the local search algorithm

        Args:
            _evaluations: {int} -- number of Local search evaluations
            
        Returns:
            {Solution} -- best solution found
        """

        # by default use of mother method to initialize variables
        super().run(_evaluations)

        # do not use here the best solution known (default use of initRun and current solution)
        # if self.parent:
        #     self.bestSolution = self.parent.bestSolution

        # initialize current solution
        self.initRun()

        solutionSize = self.currentSolution.size

        # local search algorithm implementation
        while not self.stop():

            for _ in range(solutionSize):

                # update current solution using policy
                newSolution = self.update(self.currentSolution)

                # if better solution than currently, replace it
                if self.isBetter(newSolution):
                    self.bestSolution = newSolution

                # increase number of evaluations
                self.increaseEvaluation()

                self.progress()

                logging.info("---- Current %s - SCORE %s" %
                             (newSolution, newSolution.fitness()))

                # add to surrogate pool file if necessary (using ILS parent reference)
                if self.parent.start_train_surrogate >= self.getGlobalEvaluation():
                    self.parent.add_to_surrogate(newSolution)

                # stop algorithm if necessary
                if self.stop():
                    break

            # after applying local search on currentSolution, we switch into new local area using known current bestSolution
            self.currentSolution = self.bestSolution

        logging.info("End of %s, best solution found %s" %
                     (type(self).__name__, self.bestSolution))

        return self.bestSolution

    def addCallback(self, _callback):
        """Add new callback to algorithm specifying usefull parameters

        Args:
            _callback: {Callback} -- specific Callback instance
        """
        # specify current main algorithm reference
        if self.parent is not None:
            _callback.setAlgo(self.parent)
        else:
            _callback.setAlgo(self)

        # set as new
        self.callbacks.append(_callback)
