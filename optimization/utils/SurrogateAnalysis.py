# quick object for surrogate logging data
class SurrogateAnalysis():

    def __init__(self, time, every_ls, r2_scores, r2, evaluations, n_local_search):
        self._time = time
        self._every_ls = every_ls
        self._r2_scores = r2_scores
        self._r2 = r2
        self._evaluations = evaluations
        self._n_local_search = n_local_search

