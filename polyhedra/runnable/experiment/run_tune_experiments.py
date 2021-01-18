from ray import tune


def objective(step, alpha, beta):
    return (0.1 + alpha * step / 100) ** (-1) + beta * 0.1


def _iter():
    for problem in ["bouncing_ball", "stopping_car", "cartpole"]:
        for method in ["standard", "ora"]:
            if problem == "bouncing_ball":
                for tau in [0.1, 0.05]:  # {"tau": tune.grid_search([0.1, 0.05])}
                    yield problem, method, {"tau": tau}
            elif problem == "stopping_car":
                for epsilon in [0, 0.1]:#"epsilon_input": tune.grid_search([0, 0.1])
                    yield problem, method, {"epsilon_input": epsilon}
            else:
                for tau in [0.001, 0.02, 0.005]:#"tau": tune.grid_search([0.001, 0.02, 0.005]
                    yield problem, method, {"tau": tau}


def run_parameterised_experiment(config):
    # Hyperparameters
    alpha, beta = config["alpha"], config["beta"]
    for step in range(10):
        # Iterative training function - can be any arbitrary training procedure.
        intermediate_score = objective(step, alpha, beta)
        # Feed the score back back to Tune.
        tune.report(mean_loss=intermediate_score)


analysis = tune.run(
    run_parameterised_experiment,
    name="experiment_collection",
    config={
        "main_params": tune.grid_search(list(_iter())),
        "beta": tune.choice([1, 2, 3])
    })
