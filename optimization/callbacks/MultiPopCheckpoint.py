# main imports
import os
import logging
import numpy as np

# module imports
from macop.callbacks.base import Callback
from macop.utils.progress import macop_text, macop_line


class MultiPopCheckpoint(Callback):
    """
    MultiCheckpoint is used for loading previous computations and start again after loading checkpoint

    Attributes:
        algo: {:class:`~macop.algorithms.base.Algorithm`} -- main algorithm instance reference
        every: {int} -- checkpoint frequency used (based on number of evaluations)
        filepath: {str} -- file path where checkpoints will be saved
    """
    def run(self):
        """
        Check if necessary to do backup based on `every` variable
        """
        # get current population
        population = self._algo.population

        currentEvaluation = self._algo.getGlobalEvaluation()

        # backup if necessary
        if currentEvaluation % self._every == 0:

            logging.info("Checkpoint is done into " + self._filepath)

            with open(self._filepath, 'a') as f:
                
                pop_line = str(currentEvaluation) + ';'

                scores = []
                pop_data = []

                for solution in population:
                    solution_data = ""
                    solutionSize = len(solution.data)

                    for index, val in enumerate(solution.data):
                        solution_data += str(val)

                        if index < solutionSize - 1:
                            solution_data += ' '
                    
                    scores.append(solution.fitness)
                    pop_data.append(solution_data)

                for score in scores:
                    pop_line += str(score) + ';'

                for data in pop_data:
                    pop_line += data + ';'

                pop_line += '\n'

                f.write(pop_line)

    def load(self):
        """
        Load backup lines as population and set algorithm state (population and pareto front) at this backup
        """
        if os.path.exists(self._filepath):

            logging.info('Load best solution from last checkpoint')
            with open(self._filepath, 'r') as f:

                # read data for each line
                data_line = f.readlines()[-1]
                
                data = data_line.replace(';\n', '').split(';')
          
                # get evaluation  information
                globalEvaluation = int(data[0])

                if self._algo.getParent() is not None:
                    self._algo.getParent(
                    )._numberOfEvaluations = globalEvaluation
                else:
                    self._algo._numberOfEvaluations = globalEvaluation

                nSolutions = len(self._algo.population)
                scores = list(map(float, data[1:nSolutions + 1]))

                # get best solution data information
                pop_str_data = data[nSolutions + 1:]
                pop_data = []

                for sol_data in pop_str_data:
                    current_data = list(map(int, sol_data.split(' ')))
                    pop_data.append(current_data)

                for i, sol_data in enumerate(pop_data):

                    # initialise and fill with data
                    self._algo.population[i] = self._algo.initialiser()
                    self._algo.population[i].data = np.array(sol_data)
                    self._algo.population[i].fitness = scores[i]

            macop_line(self._algo)
            macop_text(
                self._algo,
                f'Load of available population from `{self._filepath}`')
            macop_text(
                self._algo,
                f'Restart algorithm from evaluation {self._algo._numberOfEvaluations}.'
            )
        else:
            macop_text(
                self._algo,
                'No backup found... Start running algorithm from evaluation 0.'
            )
            logging.info(
                "Can't load backup... Backup filepath not valid in Checkpoint")

        macop_line(self._algo)