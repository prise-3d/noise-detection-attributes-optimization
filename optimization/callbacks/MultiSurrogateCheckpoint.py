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
        k_indices = self._algo._k_indices

        # Do nothing is surrogate analyser does not exist
        if k_indices is None:
            return

        currentEvaluation = self._algo.getGlobalEvaluation()

        # backup if necessary
        if currentEvaluation % self._every == 0:

            logging.info(f"Multi surrogate analysis checkpoint is done into {self._filepath}")

            line = str(currentEvaluation) + ';'

            for indices in k_indices:
                
                indices_data = ""
                indices_size = len(indices)

                for index, val in enumerate(indices):
                    indices_data += str(val)

                    if index < indices_size - 1:
                        indices_data += ' '

                line += indices_data + ';'

            line += '\n'

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
        if os.path.exists(self._filepath):

            logging.info('Load best solution from last checkpoint')
            with open(self._filepath) as f:

                # get last line and read data
                lastline = f.readlines()[-1]
                data = lastline.split(';')

                k_indices = data[1:]
                k_indices_final = []

                for indices in k_indices:
                    k_indices_final.append(list(map(int, indices.split(' '))))

                # set k_indices into main algorithm
                self._algo._k_indices = k_indices_final

            print(macop_line())
            print(macop_text(f' MultiSurrogateCheckpoint found from `{self._filepath}` file.'))

        else:
            print(macop_text('No backup found... Start running using new `k_indices` values'))
            logging.info("Can't load MultiSurrogate backup... Backup filepath not valid in  MultiSurrogateCheckpoint")

        print(macop_line())

