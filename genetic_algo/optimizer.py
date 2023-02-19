import warnings
from copy import copy

import numpy as np
from poap.strategy import EvalRecord
from pySOT.experimental_design import SymmetricLatinHypercube, TwoFactorial
from pySOT.optimization_problems import OptimizationProblem
from pySOT.strategy import SRBFStrategy
from pySOT.surrogate import CubicKernel, LinearTail, RBFInterpolant, SurrogateUnitBox

from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main
from bayesmark.space import JointSpace


from pypop7.optimizers.es.lmmaes import LMMAES

from pypop7.optimizers.ga.asga import ASGA


class PySOTOptimizer(AbstractOptimizer):
    primary_import = "pysot"

    def __init__(self, api_config):
        """Build wrapper class to use an optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)

        self.space_x = JointSpace(api_config)
        self.bounds = self.space_x.get_bounds()
        self.create_opt_prob()  # Sets up the optimization problem (needs self.bounds)
        self.max_evals = np.iinfo(np.int32).max  # NOTE: Largest possible int
        self.batch_size = None
        self.history = []
        self.proposals = []


    def create_opt_prob(self):

        problem = {'fitness_function': None,  # cost function
           'ndim_problem': len(self.bounds),  # dimension
           'lower_boundary': self.bounds[:, 0],  # search boundary
           'upper_boundary': self.bounds[:, 1]}

        self.problem = problem


    def start(self, max_evals):
        """Starts a new pySOT run."""
        self.history = []
        self.proposals = []


        options = {'fitness_threshold': 1e-10,  # terminate when the best-so-far fitness is lower than this threshold
           'max_runtime': 10,  # 1 hours (terminate when the actual runtime exceeds it)
           'seed_rng': 0,  # seed of random number generation (which must be explicitly set for repeatability)
           'x': 4 * np.ones((ndim_problem,)),  # initial mean of search (mutation/sampling) distribution
           'sigma': 0.3,  # initial global step-size of search distribution
           'verbose': 500}

        self.strategy = ASGA(self.problem, options)

        self.x, self.y, self.x_as, self.y_as = self.strategy.initialize()



    def suggest(self, n_suggestions=1):
        """Get a suggestion from the optimizer.

        Parameters
        ----------
        n_suggestions : int
            Desired number of parallel suggestions in the output

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """



        if self.batch_size is None:  # First call to suggest
            self.batch_size = n_suggestions
            self.start(self.max_evals)


        for _ in range(n_suggestions):
            try:
                self.x, self.y, self.x_as, self.y_as = self.strategy.iterate(self.x, self.y, self.x_as, self.y_as, None)
                self.strategy._n_generations += 1
            except np.linalg.LinAlgError:
                self.x, self.y, self.x_as, self.y_as = self.strategy.initialize(None)
            except ValueError:
                self.x, self.y, self.x_as, self.y_as = self.strategy.initialize(None)


        return self.y


    def observe(self, X, y):
        """Send an observation of a suggestion back to the optimizer.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """


if __name__ == "__main__":
    experiment_main(PySOTOptimizer)
