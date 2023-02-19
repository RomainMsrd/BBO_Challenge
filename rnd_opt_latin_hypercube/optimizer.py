# A random search optimizer with Latin hypercube sampling.

from bayesmark import np_util
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main

class RandomOptimizer(AbstractOptimizer):
    # Unclear what is best package to list for primary_import here.
    primary_import = "bayesmark"

    def __init__(self, api_config, random=np_util.random):
        """Build wrapper class to use random search function in benchmark.

        Settings for `suggest_dict` can be passed using kwargs.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)
        self.random = random

    def suggest(self, n_suggestions=1):
        """Get suggestion.

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
        # Use Latin hypercube sampling to get diverse points
        bounds, types, _ = self.api_config.get_api_config()
        x_guess = np_util.lhs(len(bounds), n_suggestions,
                              seed=self.random.randint(1, 10000))
        x_guess = np_util.scale_to_bounds(x_guess, bounds)

        # Convert to dict format and return
        return np_util.vector_to_params(x_guess, types)

        
    def observe(self, X, y):
        """Feed an observation back.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        # Random search so don't do anything
        pass


if __name__ == "__main__":
    experiment_main(RandomOptimizer)
