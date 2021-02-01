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

        # copy data of solution
        firstData = solution1.data.copy()

        population = self._algo.population if self._algo.population is not None else self._algo.getParent().population

        # copy of solution2 as output solution
        valid = False
        copy_solution = None

        # use of different random population solution
        ncounter = 0

        while not valid:

            chosen_solution = population[random.randint(0, len(population) - 1)]
            
            if not list(chosen_solution.data) == list(firstData) or ncounter > 10:
                valid = True
                copy_solution = chosen_solution.clone()

            # add security
            ncounter += 1

        # default empty solution
        if copy_solution is None:
            copy_solution = self._algo.initialiser()

        # random split index
        splitIndex = int(size / 2)

        if random.uniform(0, 1) > 0.5:
            copy_solution.data[splitIndex:] = firstData[splitIndex:]
        else:
            copy_solution.data[:splitIndex] = firstData[:splitIndex]

        return copy_solution


class RandomPopCrossover(Crossover):

    def apply(self, solution1, solution2=None):
        """Create new solution based on best solution found and solution passed as parameter

        Args:
            solution1: {:class:`~macop.solutions.base.Solution`} -- the first solution to use for generating new solution
            solution2: {:class:`~macop.solutions.base.Solution`} -- the second solution to use for generating new solution (using population)

        Returns:
            {:class:`~macop.solutions.base.Solution`}: new generated solution
        """

        size = solution1._size

        # copy data of solution
        firstData = solution1.data.copy()

        population = self._algo.population if self._algo.population is not None else self._algo.getParent().population

        # copy of solution2 as output solution
        valid = False
        copy_solution = None

        # use of different random population solution
        ncounter = 0

        while not valid:

            chosen_solution = population[random.randint(0, len(population) - 1)]
            
            if not list(chosen_solution.data) == list(firstData) or ncounter > 10:
                valid = True
                copy_solution = chosen_solution.clone()

            # add security
            ncounter += 1

        # default empty solution
        if copy_solution is None:
            copy_solution = self._algo.initialiser()

        # random split index
        splitIndex = random.randint(0, len(population) - 1)

        if random.uniform(0, 1) > 0.5:
            copy_solution.data[splitIndex:] = firstData[splitIndex:]
        else:
            copy_solution.data[:splitIndex] = firstData[:splitIndex]

        return copy_solution