"""Basic Checkpoint class implementation
"""

# main imports
import os
import logging
import numpy as np

# module imports
from macop.callbacks.Callback import Callback
from macop.utils.color import macop_text, macop_line


class MultiSurrogateSpecificCheckpoint(Callback):
    """
    MultiSurrogateSpecificCheckpoint is used for keep track of sub-surrogate problem indices

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
        population = self._algo._population

        # Do nothing is surrogate analyser does not exist
        if population is None:
            return

        currentEvaluation = self._algo.getGlobalEvaluation()

        # backup if necessary
        if currentEvaluation % self._every == 0:

            logging.info(f"Multi surrogate specific analysis checkpoint is done into {self._filepath}")

            line = ''

            fitness_list = [ s.fitness for s in population ]
            fitness_data = ' '.join(list(map(str, fitness_list)))

            for s in population:
                s_data = ' '.join(list(map(str, s._data)))
                line += s_data + ';'

            line += fitness_data

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
        Load previous population
        """
        if os.path.exists(self._filepath):

            logging.info('Load population solutions from last checkpoint')
            with open(self._filepath) as f:

                # get last line and read data
                lastline = f.readlines()[-1].replace('\n', '')
                data = lastline.split(';')

                fitness_scores = list(map(float, data[-1].split(' ')))

                for i, solution_data in enumerate(data[:-1]):
                    self._algo._population[i]._data = list(map(int, solution_data.split(' ')))
                    self._algo._population[i]._score = fitness_scores[i]

            print(macop_line())
            print(macop_text(f' MultiSurrogateSpecificCheckpoint found from `{self._filepath}` file. Start running using previous `population` values'))

            for i, s in enumerate(self._algo._population):
                print(f'Population[{i}]: best solution fitness is {s.fitness}')

        else:
            print(macop_text('No backup found... Start running using new `population` values'))
            logging.info("Can't load MultiSurrogateSpecific backup... Backup filepath not valid in  MultiSurrogateCheckpoint")

        print(macop_line())

