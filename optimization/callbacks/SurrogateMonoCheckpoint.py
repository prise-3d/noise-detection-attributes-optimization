"""Basic Checkpoint class implementation
"""

# main imports
import os
import logging
import numpy as np

# module imports
from macop.callbacks.base import Callback
from macop.utils.progress import macop_text, macop_line


class SurrogateMonoCheckpoint(Callback):
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

            # get score of r² and mae

            line = str(currentEvaluation) + ';' + str(surrogate_analyser._n_local_search) + ';' + str(surrogate_analyser._every_ls) + ';' + str(surrogate_analyser._time)  + ';' + str(surrogate_analyser._r2) \
                + ';' + str(surrogate_analyser._mae) \
                + ';' + solutionData + ';' + str(solution.fitness) + ';\n'

            # check if file exists
            if not os.path.exists(self._filepath):
                with open(self._filepath, 'w') as f:
                    f.write(line)
            else:
                with open(self._filepath, 'a') as f:
                    f.write(line)

    def load(self):
        """
        only load global n local search
        """

        if os.path.exists(self._filepath):

            logging.info('Load n local search')
            with open(self._filepath) as f:

                # get last line and read data
                lastline = f.readlines()[-1].replace(';\n', '')
                data = lastline.split(';')

                n_local_search = int(data[1])

                # set k_indices into main algorithm
                self._algo._total_n_local_search = n_local_search

            print(macop_line())
            print(macop_text(f'SurrogateMonoCheckpoint found from `{self._filepath}` file.'))

        else:
            print(macop_text('No backup found...'))
            logging.info("Can't load Surrogate backup... Backup filepath not valid in SurrogateCheckpoint")

        print(macop_line())
