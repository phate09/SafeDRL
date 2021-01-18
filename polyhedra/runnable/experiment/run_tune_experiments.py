import ray
from ray import tune

from polyhedra.runnable.experiment.run_experiment_bouncing_ball import BouncingBallExperiment
from polyhedra.runnable.experiment.run_experiment_cartpole import CartpoleExperiment
from polyhedra.runnable.experiment.run_experiment_stopping_car import StoppingCarExperiment


def _iter():
    for problem in ["bouncing_ball"]:  # , "stopping_car", "cartpole"
        for method in ["standard"]:  # , "ora"
            if problem == "bouncing_ball":
                for tau in [0.1]:  # {"tau": tune.grid_search([0.1, 0.05])}
                    nn_paths = ["/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_c7326_00000_0_2021-01-16_05-43-36/checkpoint_36/checkpoint-36",
                                "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_bd56c_00000_0_tau=0.1_2021-01-17_14-46-10/checkpoint_114/checkpoint-114",
                                "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_bd56c_00001_1_tau=0.05_2021-01-17_14-46-10/checkpoint_457/checkpoint-457",
                                "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_bd56c_00002_2_tau=0.1_2021-01-17_14-55-31/checkpoint_223/checkpoint-223",
                                "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_bd56c_00003_3_tau=0.05_2021-01-17_15-13-50/checkpoint_139/checkpoint-139",
                                "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_bd56c_00004_4_tau=0.1_2021-01-17_15-15-24/checkpoint_297/checkpoint-297",
                                "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_bd56c_00005_5_tau=0.05_2021-01-17_15-23-02/checkpoint_925/checkpoint-925",
                                "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_bd56c_00006_6_tau=0.1_2021-01-17_15-38-38/checkpoint_87/checkpoint-87",
                                "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_bd56c_00007_7_tau=0.05_2021-01-17_15-45-25/checkpoint_69/checkpoint-69",
                                "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_bd56c_00008_8_tau=0.1_2021-01-17_15-50-09/checkpoint_83/checkpoint-83",
                                "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_bd56c_00009_9_tau=0.05_2021-01-17_15-57-00/checkpoint_897/checkpoint-897",
                                "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_bd56c_00010_10_tau=0.1_2021-01-17_16-21-05/checkpoint_234/checkpoint-234",
                                "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_bd56c_00011_11_tau=0.05_2021-01-17_16-40-13/checkpoint_897/checkpoint-897",
                                "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_bd56c_00012_12_tau=0.1_2021-01-17_16-55-18/checkpoint_250/checkpoint-250",
                                "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_bd56c_00013_13_tau=0.05_2021-01-17_17-14-35/checkpoint_447/checkpoint-447",
                                "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_bd56c_00014_14_tau=0.1_2021-01-17_17-36-01/checkpoint_310/checkpoint-310",
                                "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_bd56c_00015_15_tau=0.05_2021-01-17_17-41-56/checkpoint_898/checkpoint-898",
                                "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_bd56c_00016_16_tau=0.1_2021-01-17_18-00-09/checkpoint_157/checkpoint-157",
                                "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_bd56c_00017_17_tau=0.05_2021-01-17_18-12-21/checkpoint_899/checkpoint-899",
                                "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_bd56c_00018_18_tau=0.1_2021-01-17_18-36-24/checkpoint_699/checkpoint-699",
                                "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_bd56c_00019_19_tau=0.05_2021-01-17_19-08-54/checkpoint_961/checkpoint-961"]
                    for nn_path in nn_paths:
                        yield problem, method, {"tau": tau, "nn_path": nn_path}
            elif problem == "stopping_car":
                for epsilon in [0, 0.1]:  # "epsilon_input": tune.grid_search([0, 0.1])
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
                        yield problem, method, {"epsilon_input": epsilon, "nn_path": nn_path}
            else:
                for tau in [0.001, 0.02, 0.005]:  # "tau": tune.grid_search([0.001, 0.02, 0.005]
                    nn_paths = ["/home/edoardo/ray_results/tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00000_0_cost_fn=0,tau=0.001_2021-01-16_20-25-43/checkpoint_193/checkpoint-193",
                                "/home/edoardo/ray_results/tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00001_1_cost_fn=1,tau=0.001_2021-01-16_20-25-43/checkpoint_3334/checkpoint-3334",
                                "/home/edoardo/ray_results/tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00002_2_cost_fn=2,tau=0.001_2021-01-16_20-33-36/checkpoint_3334/checkpoint-3334",
                                "/home/edoardo/ray_results/tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00003_3_cost_fn=0,tau=0.02_2021-01-16_23-08-42/checkpoint_190/checkpoint-190",
                                "/home/edoardo/ray_results/tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00004_4_cost_fn=1,tau=0.02_2021-01-16_23-14-15/checkpoint_3334/checkpoint-3334",
                                "/home/edoardo/ray_results/tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00005_5_cost_fn=2,tau=0.02_2021-01-16_23-27-15/checkpoint_3334/checkpoint-3334"]
                    for nn_path in nn_paths:
                        yield problem, method, {"tau": tau, "nn_path": nn_path}


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
        experiment.n_workers = n_workers
        experiment.show_progressbar = False
        experiment.save_dir = trial_dir
        experiment.update_progress_fn = update_progress
        elapsed_seconds, safe, max_t = experiment.run_experiment()
    elif problem == "stopping_car":
        experiment = StoppingCarExperiment()
        experiment.nn_path = other_config["nn_path"]
        experiment.input_epsilon = other_config["epsilon"]
        experiment.n_workers = n_workers
        experiment.show_progressbar = False
        experiment.save_dir = trial_dir
        experiment.update_progress_fn = update_progress
        elapsed_seconds, safe, max_t = experiment.run_experiment()
    else:
        experiment = CartpoleExperiment()
        experiment.nn_path = other_config["nn_path"]
        experiment.tau = other_config["tau"]
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
