import activations.sigmoid_approx as s
from ray import tune
import hyperopt as hp
from ray.tune.suggest.hyperopt import HyperOptSearch
import torch
import plotly.graph_objects as go


def objective_over(shift, slope, min_val):
    input = torch.linspace(-10, 10, steps=10000)
    sig = torch.nn.Sigmoid()
    over_approx = s.sigmoid_approx(shift=shift, slope=slope, min_val=min_val)
    result_over = over_approx(input)
    result_sigmoid = sig(input)
    result_area = torch.stack([torch.abs(x - y) if x > y else torch.abs(x - y) * 1000 for x, y in zip(result_over, result_sigmoid)])
    return torch.sum(result_area).item()


def objective_under(shift, slope, max_val):
    input = torch.linspace(-10, 10, steps=10000)
    sig = torch.nn.Sigmoid()
    under_approx = s.sigmoid_approx(shift=shift, slope=slope, max_val=max_val)
    result_under = under_approx(input)
    result_sigmoid = sig(input)
    result_area = torch.stack([torch.abs(x - y) if x < y else torch.abs(x - y) * 1000  for x, y in zip(result_under, result_sigmoid)]) #
    return torch.sum(result_area).item()


def objective(x, a, b):
    return a * (x ** 0.5) + b


def trainable(config):
    # config (dict): A dict of hyperparameters.
    underapprox = True

    if underapprox:
        score = objective_under(shift=config["shift"], slope=config["slope"], max_val=config["max_val"])
    else:
        score = objective_over(shift=config["shift"], slope=config["slope"], min_val=config["min_val"])
    tune.report(score=score)  # This sends the score to Tune.


def test_sigmoid(config):
    input = torch.linspace(-10, 10, steps=10000)
    sig = torch.nn.Sigmoid()
    under_approx = s.sigmoid_approx(shift=config["shift"], slope=config["slope"], max_val=config["max_val"])
    result_under = under_approx(input)
    result_sigmoid = sig(input)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=input.numpy(), y=result_under.numpy(), mode='lines+markers', name='under_approx'))
    fig.add_trace(go.Scatter(x=input.numpy(), y=result_sigmoid.numpy(), mode='lines+markers', name='sigmoid'))
    fig.show()


if __name__ == '__main__':

    space = {"shift": hp.hp.uniform("shift", -0.5, 0.5), "slope": hp.hp.uniform("slope", 0.1, 0.3), "max_val": hp.hp.uniform("max_val", 0.98, 0.99)}
    old_params =  { "shift":0,"slope":0.2,"max_val":0.99}
    hyperopt = HyperOptSearch(space, metric="score", mode="min",points_to_evaluate=[old_params])
    analysis = tune.run(trainable, search_alg=hyperopt, num_samples=1000)
    # Get the best hyperparameters
    best_hyperparameters = analysis.get_best_config("score")
    print(f"best hyperparameters {best_hyperparameters}")
    test_sigmoid(best_hyperparameters)
    end_score = objective_under(shift=best_hyperparameters["shift"], slope=best_hyperparameters["slope"], max_val=best_hyperparameters["max_val"])
    print(end_score)
    print(end_score)
