"""Basic Checkpoint class implementation
"""

# main imports
import os
import logging
import numpy as np

# module imports
from macop.callbacks.Callback import Callback
from macop.utils.color import macop_text, macop_line


class SurrogateCheckpoint(Callback):
    """
    SurrogateCheckpoint is used for logging training data information about surrogate

    Attributes:
        algo: {Algorithm} -- main algorithm instance reference
        every: {int} -- checkpoint frequency used (based on number of evaluations)
        filepath: {str} -- file path where checkpoints will be saved
    """
    def run(self):
        """
        Check if necessary to do backup based on `every` variable
        """
        # get current best solution
        solution = self._algo._bestSolution
        surrogate_analyser = self._algo._surrogate_analyser

        # Do nothing is surrogate analyser does not exist
        if surrogate_analyser is None:
            return

        currentEvaluation = self._algo.getGlobalEvaluation()

        # backup if necessary
        if currentEvaluation % self._every == 0:

            logging.info(f"Surrogate analysis checkpoint is done into {self._filepath}")

            solutionData = ""
            solutionSize = len(solution._data)

            for index, val in enumerate(solution._data):
                solutionData += str(val)

                if index < solutionSize - 1:
                    solutionData += ' '

            line = str(currentEvaluation) + ';' + str(surrogate_analyser._every_ls) + ';' + str(surrogate_analyser._time) + ';' + str(surrogate_analyser._r2) \
                + ';' + solutionData + ';' + str(solution.fitness()) + ';\n'

            # check if file exists
            if not os.path.exists(self._filepath):
                with open(self._filepath, 'w') as f:
                    f.write(line)
            else:
                with open(self._filepath, 'a') as f:
                    f.write(line)

    def load(self):
        """
        Load nothing there, as we only log surrogate training information
        """

        logging.info("No loading to do with surrogate checkpoint")
