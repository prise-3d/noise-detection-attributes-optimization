from macop.operators.base import Crossover
import random

class SimplePopCrossover(Crossover):

    def apply(self, solution1, solution2=None):
        """Create new solution based on best solution found and solution passed as parameter

        Args:
            solution1: {:class:`~macop.solutions.base.Solution`} -- the first solution to use for generating new solution
            solution2: {:class:`~macop.solutions.base.Solution`} -- the second solution to use for generating new solution (using population)

        Returns:
            {:class:`~macop.solutions.base.Solution`}: new generated solution
        """

        size = solution1._size
        population = self._algo.population

        # copy data of solution
        firstData = solution1.data.copy()

        # copy of solution2 as output solution
        valid = False
        copy_solution = None

        # use of different random population solution
        ncounter = 0
        while not valid:

            chosen_solution = population[random.randint(0, len(population))]
            
            if chosen_solution.data != firstData or ncounter > 10:
                valid = True
                copy_solution = chosen_solution.clone()

            # add security
            ncounter += 1

        splitIndex = int(size / 2)

        if random.uniform(0, 1) > 0.5:
            copy_solution.data[splitIndex:] = firstData[splitIndex:]
        else:
            copy_solution.data[:splitIndex] = firstData[:splitIndex]

        return copy_solution