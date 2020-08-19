import activations.sigmoid_approx as sigmoid_approx
from ray import tune
import hyperopt as hp
from ray.tune.suggest.hyperopt import HyperOptSearch


def objective_over():
    pass


def objective(x, a, b):
    return a * (x ** 0.5) + b


# Create a HyperOpt search space

def trainable(config):
    # config (dict): A dict of hyperparameters.

    for x in range(20):
        score = objective(x, config["a"], config["b"])

        tune.report(score=score)  # This sends the score to Tune.


if __name__ == '__main__':
    space = {"a": hp.hp.uniform("a", 0, 1), "b": hp.hp.uniform("b", 0, 20)

             # Note: Arbitrary HyperOpt search spaces should be supported!
             # "foo": hp.lognormal("foo", 0, 1))
             }
    hyperopt = HyperOptSearch(space, metric="score", mode="max")
    analysis = tune.run(trainable, search_alg=hyperopt, num_samples=20, stop={"training_iteration": 20})
    # Get the best hyperparameters
    best_hyperparameters = analysis.get_best_config("score")
    print(f"best hyperparameters {best_hyperparameters}")
