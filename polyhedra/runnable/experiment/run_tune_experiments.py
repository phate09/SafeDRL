import ray
from ray import tune

from polyhedra.experiments_nn_analysis import Experiment
from polyhedra.runnable.experiment.run_experiment_bouncing_ball import BouncingBallExperiment
from polyhedra.runnable.experiment.run_experiment_cartpole import CartpoleExperiment
from polyhedra.runnable.experiment.run_experiment_stopping_car import StoppingCarExperiment


def _iter():
    for problem in ["bouncing_ball", "stopping_car", "cartpole"]:  #
        for method in ["standard"]:  # , "ora"
            if problem == "bouncing_ball":
                for tau in [0.1]:  # {"tau": tune.grid_search([0.1, 0.05])}
                    for template in [0, 1, 2]:
                        nn_paths = [
                            "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_c7326_00000_0_2021-01-16_05-43-36/checkpoint_36/checkpoint-36",
                            "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_71684_00000_0_2021-01-18_23-46-54/checkpoint_10/checkpoint-10",
                            "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_71684_00001_1_2021-01-18_23-46-54/checkpoint_10/checkpoint-10",
                            "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_71684_00002_2_2021-01-18_23-47-37/checkpoint_10/checkpoint-10",
                            "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_71684_00003_3_2021-01-18_23-47-37/checkpoint_10/checkpoint-10",
                            "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_71684_00004_4_2021-01-18_23-48-21/checkpoint_10/checkpoint-10",
                            "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_fb929_00000_0_2021-01-19_00-19-23/checkpoint_10/checkpoint-10",
                            "//home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_fb929_00001_1_2021-01-19_00-19-23/checkpoint_20/checkpoint-20",
                            "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_fb929_00002_2_2021-01-19_00-20-06/checkpoint_10/checkpoint-10",
                            "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_fb929_00003_3_2021-01-19_00-20-45/checkpoint_10/checkpoint-10",
                            "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_fb929_00004_4_2021-01-19_00-20-52/checkpoint_20/checkpoint-20",
                        ]
                        for nn_path in nn_paths:
                            yield problem, method, {"tau": tau, "template": template, "nn_path": nn_path}
            elif problem == "stopping_car":
                for epsilon in [0, 0.1]:  # "epsilon_input": tune.grid_search([0, 0.1])
                    for template in [0, 1, 2]:
                        nn_paths = ["/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_14b68_00000_0_cost_fn=0,epsilon_input=0_2021-01-17_11-56-58/checkpoint_31/checkpoint-31",
                                    "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_14b68_00001_1_cost_fn=0,epsilon_input=0.1_2021-01-17_11-56-58/checkpoint_37/checkpoint-37",
                                    "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00000_0_cost_fn=0,epsilon_input=0_2021-01-17_12-37-27/checkpoint_24/checkpoint-24",
                                    "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00001_1_cost_fn=0,epsilon_input=0.1_2021-01-17_12-37-27/checkpoint_36/checkpoint-36",
                                    "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00002_2_cost_fn=0,epsilon_input=0_2021-01-17_12-38-53/checkpoint_40/checkpoint-40",
                                    "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00003_3_cost_fn=0,epsilon_input=0.1_2021-01-17_12-39-31/checkpoint_32/checkpoint-32",
                                    "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00004_4_cost_fn=0,epsilon_input=0_2021-01-17_12-41-14/checkpoint_76/checkpoint-76",
                                    "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00005_5_cost_fn=0,epsilon_input=0.1_2021-01-17_12-41-27/checkpoint_58/checkpoint-58",
                                    "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00006_6_cost_fn=0,epsilon_input=0_2021-01-17_12-44-54/checkpoint_41/checkpoint-41",
                                    "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00007_7_cost_fn=0,epsilon_input=0.1_2021-01-17_12-45-46/checkpoint_89/checkpoint-89",
                                    "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00008_8_cost_fn=0,epsilon_input=0_2021-01-17_12-47-19/checkpoint_43/checkpoint-43",
                                    "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00009_9_cost_fn=0,epsilon_input=0.1_2021-01-17_12-49-48/checkpoint_50/checkpoint-50",
                                    "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00010_10_cost_fn=0,epsilon_input=0_2021-01-17_12-51-01/checkpoint_27/checkpoint-27",
                                    "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00011_11_cost_fn=0,epsilon_input=0.1_2021-01-17_12-52-36/checkpoint_44/checkpoint-44",
                                    "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00012_12_cost_fn=0,epsilon_input=0_2021-01-17_12-52-47/checkpoint_50/checkpoint-50",
                                    "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00013_13_cost_fn=0,epsilon_input=0.1_2021-01-17_12-55-12/checkpoint_53/checkpoint-53",
                                    "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00014_14_cost_fn=0,epsilon_input=0_2021-01-17_12-55-46/checkpoint_56/checkpoint-56",
                                    "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00015_15_cost_fn=0,epsilon_input=0.1_2021-01-17_12-58-23/checkpoint_48/checkpoint-48",
                                    "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00016_16_cost_fn=0,epsilon_input=0_2021-01-17_12-59-01/checkpoint_38/checkpoint-38",
                                    "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00017_17_cost_fn=0,epsilon_input=0.1_2021-01-17_13-01-15/checkpoint_50/checkpoint-50",
                                    "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00018_18_cost_fn=0,epsilon_input=0_2021-01-17_13-01-17/checkpoint_35/checkpoint-35",
                                    "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00019_19_cost_fn=0,epsilon_input=0.1_2021-01-17_13-03-22/checkpoint_36/checkpoint-36"]
                        for nn_path in nn_paths:
                            yield problem, method, {"epsilon_input": epsilon, "template": template, "nn_path": nn_path}
            else:
                for tau in [0.001, 0.02, 0.005]:  # "tau": tune.grid_search([0.001, 0.02, 0.005]
                    for template in [0, 1, 2]:
                        nn_paths = ["/home/edoardo/ray_results/tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00000_0_cost_fn=0,tau=0.001_2021-01-16_20-25-43/checkpoint_193/checkpoint-193",
                                    "/home/edoardo/ray_results/tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00001_1_cost_fn=1,tau=0.001_2021-01-16_20-25-43/checkpoint_3334/checkpoint-3334",
                                    "/home/edoardo/ray_results/tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00002_2_cost_fn=2,tau=0.001_2021-01-16_20-33-36/checkpoint_3334/checkpoint-3334",
                                    "/home/edoardo/ray_results/tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00003_3_cost_fn=0,tau=0.02_2021-01-16_23-08-42/checkpoint_190/checkpoint-190",
                                    "/home/edoardo/ray_results/tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00004_4_cost_fn=1,tau=0.02_2021-01-16_23-14-15/checkpoint_3334/checkpoint-3334",
                                    "/home/edoardo/ray_results/tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00005_5_cost_fn=2,tau=0.02_2021-01-16_23-27-15/checkpoint_3334/checkpoint-3334"]
                        for nn_path in nn_paths:
                            yield problem, method, {"tau": tau, "template": template, "nn_path": nn_path}


def update_progress(n_workers, seen, frontier, num_already_visited, max_t):
    tune.report(n_workers=n_workers, seen=seen, frontier=frontier, num_already_visited=num_already_visited, max_t=max_t)


def run_parameterised_experiment(config):
    # Hyperparameters
    trial_dir = tune.get_trial_dir()
    problem, method, other_config = config["main_params"]
    n_workers = config["n_workers"]
    if problem == "bouncing_ball":
        experiment = BouncingBallExperiment()
        experiment.nn_path = other_config["nn_path"]
        experiment.tau = other_config["tau"]
        if other_config["template"] == 2:  # octagon
            experiment.analysis_template = Experiment.octagon(experiment.env_input_size)
        elif other_config["template"] == 0:  # box
            experiment.analysis_template = Experiment.box(experiment.env_input_size)
        else:
            experiment.analysis_template = experiment.get_template(1)  # standard
        experiment.n_workers = n_workers
        experiment.show_progressbar = False
        experiment.save_dir = trial_dir
        experiment.update_progress_fn = update_progress
        elapsed_seconds, safe, max_t = experiment.run_experiment()
    elif problem == "stopping_car":
        experiment = StoppingCarExperiment()
        experiment.nn_path = other_config["nn_path"]
        experiment.input_epsilon = other_config["epsilon_input"]
        if other_config["template"] == 2:  # octagon
            experiment.analysis_template = Experiment.octagon(experiment.env_input_size)
        elif other_config["template"] == 0:  # box
            experiment.analysis_template = Experiment.box(experiment.env_input_size)
        else:
            experiment.analysis_template = experiment.get_template(1)  # standard
        experiment.n_workers = n_workers
        experiment.show_progressbar = False
        experiment.save_dir = trial_dir
        experiment.update_progress_fn = update_progress
        elapsed_seconds, safe, max_t = experiment.run_experiment()
    else:
        experiment = CartpoleExperiment()
        experiment.nn_path = other_config["nn_path"]
        experiment.tau = other_config["tau"]
        if other_config["template"] == 2:  # octagon
            experiment.analysis_template = Experiment.octagon(experiment.env_input_size)
        elif other_config["template"] == 0:  # box
            experiment.analysis_template = Experiment.box(experiment.env_input_size)
        else:
            experiment.analysis_template = experiment.get_template(1)  # standard
        experiment.n_workers = n_workers
        experiment.show_progressbar = False
        experiment.save_dir = trial_dir
        experiment.update_progress_fn = update_progress
        elapsed_seconds, safe, max_t = experiment.run_experiment()
    safe_value = 0
    if safe is None:
        safe_value = 0
    elif safe:
        safe_value = 1
    elif not safe:
        safe_value = -1
    tune.report(elapsed_seconds=elapsed_seconds, safe=safe_value, max_t=max_t, done=True)


if __name__ == '__main__':
    ray.init(local_mode=True)
    cpu = 7
    analysis = tune.run(
        run_parameterised_experiment,
        name="experiment_collection",
        config={
            "main_params": tune.grid_search(list(_iter())),
            "n_workers": cpu
        },
        resources_per_trial={"cpu": cpu},
        log_to_file=True)
    df = analysis.results_df
    df.to_json("~/ray_results/experiment_collection/experiment_results.json")
