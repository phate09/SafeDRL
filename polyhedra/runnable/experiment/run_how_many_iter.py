import datetime
import os
import time

import ray
from ray import tune
import re
from polyhedra.experiments_nn_analysis import Experiment
from polyhedra.runnable.experiment.run_experiment_bouncing_ball import BouncingBallExperiment
from polyhedra.runnable.experiment.run_experiment_cartpole import CartpoleExperiment
from polyhedra.runnable.experiment.run_experiment_stopping_car import StoppingCarExperiment

nn_paths_cartpole = ["/home/edoardo/ray_results/tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00000_0_cost_fn=0,tau=0.001_2021-01-16_20-25-43/checkpoint_193/checkpoint-193",
                     "/home/edoardo/ray_results/tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00001_1_cost_fn=1,tau=0.001_2021-01-16_20-25-43/checkpoint_3334/checkpoint-3334",
                     "/home/edoardo/ray_results/tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00002_2_cost_fn=2,tau=0.001_2021-01-16_20-33-36/checkpoint_3334/checkpoint-3334",
                     "/home/edoardo/ray_results/tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00003_3_cost_fn=0,tau=0.02_2021-01-16_23-08-42/checkpoint_190/checkpoint-190",
                     "/home/edoardo/ray_results/tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00004_4_cost_fn=1,tau=0.02_2021-01-16_23-14-15/checkpoint_3334/checkpoint-3334",
                     "/home/edoardo/ray_results/tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00005_5_cost_fn=2,tau=0.02_2021-01-16_23-27-15/checkpoint_3334/checkpoint-3334"]

folder_cartpole = [

    "/home/edoardo/ray_results/tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00000_0_cost_fn=0,tau=0.001_2021-01-16_20-25-43",
    "/home/edoardo/ray_results/tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00001_1_cost_fn=1,tau=0.001_2021-01-16_20-25-43"
    # "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00000_0_cost_fn=0,epsilon_input=0_2021-01-17_12-37-27",
    # "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00001_1_cost_fn=0,epsilon_input=0.1_2021-01-17_12-37-27",
    # "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00002_2_cost_fn=0,epsilon_input=0_2021-01-17_12-38-53",
    # "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00003_3_cost_fn=0,epsilon_input=0.1_2021-01-17_12-39-31",
    # "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00004_4_cost_fn=0,epsilon_input=0_2021-01-17_12-41-14",  # safe at checkpoint 10
    # "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00005_5_cost_fn=0,epsilon_input=0.1_2021-01-17_12-41-27",
    # "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00006_6_cost_fn=0,epsilon_input=0_2021-01-17_12-44-54",
    # "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00007_7_cost_fn=0,epsilon_input=0.1_2021-01-17_12-45-46",
    # "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00008_8_cost_fn=0,epsilon_input=0_2021-01-17_12-47-19",
    # "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00009_9_cost_fn=0,epsilon_input=0.1_2021-01-17_12-49-48",
    # "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00010_10_cost_fn=0,epsilon_input=0_2021-01-17_12-51-01",
    # "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00011_11_cost_fn=0,epsilon_input=0.1_2021-01-17_12-52-36",
    # "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00012_12_cost_fn=0,epsilon_input=0_2021-01-17_12-52-47",
    # "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00013_13_cost_fn=0,epsilon_input=0.1_2021-01-17_12-55-12",
    # "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00014_14_cost_fn=0,epsilon_input=0_2021-01-17_12-55-46",
    # "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00015_15_cost_fn=0,epsilon_input=0.1_2021-01-17_12-58-23",
    # "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00016_16_cost_fn=0,epsilon_input=0_2021-01-17_12-59-01",
    # "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00017_17_cost_fn=0,epsilon_input=0.1_2021-01-17_13-01-15",
    # "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00018_18_cost_fn=0,epsilon_input=0_2021-01-17_13-01-17",
    # "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00019_19_cost_fn=0,epsilon_input=0.1_2021-01-17_13-03-22"

]


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def find_checkpoints(folder):
    list_subfolders_with_paths = [f.name for f in os.scandir(folder) if f.is_dir()]
    if "_skip" in list_subfolders_with_paths:
        list_subfolders_with_paths.remove("_skip")
    list_subfolders_with_paths.sort(key=natural_keys)
    result = [(x, os.path.join(os.path.join(folder, x), x.replace("_", "-"))) for x in list_subfolders_with_paths]
    return result


def _iter():
    for problem in ["cartpole"]:  # "bouncing_ball", "stopping_car", "cartpole"
        for method in ["standard"]:  # , "ora"
            if problem == "cartpole":
                for tau in [0.001]:  # "tau": tune.grid_search([0.001, 0.02, 0.005]
                    for template in [1]:  # [1,0,2]
                        for nn_index in range(1, min(100, len(folder_cartpole))):
                            for checkpoint, check_folder in find_checkpoints(folder_cartpole[nn_index]):
                                yield problem, method, {"tau": tau, "template": template, "agent": nn_index, "checkpoint": checkpoint, "folder": check_folder}


def update_progress(n_workers, seen, frontier, num_already_visited, max_t):
    tune.report(n_workers=n_workers, seen=seen, frontier=frontier, num_already_visited=num_already_visited, max_t=max_t)


def run_parameterised_experiment(config):
    # Hyperparameters
    trial_dir = tune.get_trial_dir()
    problem, method, other_config = config["main_params"]
    n_workers = config["n_workers"]

    experiment = CartpoleExperiment()
    experiment.nn_path = other_config["folder"]  # nn_paths_cartpole[other_config["nn_path"]]
    experiment.tau = other_config["tau"]
    if other_config["template"] == 2:  # octagon
        experiment.analysis_template = Experiment.octagon(experiment.env_input_size)
    elif other_config["template"] == 0:  # box
        experiment.analysis_template = Experiment.box(experiment.env_input_size)
    else:
        _, template = experiment.get_template(1)
        experiment.analysis_template = template  # standard
    experiment.n_workers = n_workers
    experiment.show_progressbar = False
    experiment.show_progress_plot = False
    # experiment.use_rounding = False
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


class NameGroup:
    def __init__(self):
        ts = int(time.time())
        d = datetime.datetime.fromtimestamp(ts)
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.date_time = datetime_str
        self.name = hex(int(time.time()))[2:]

    def trial_str_creator(self, trial):
        problem, method, other_config = trial.config["main_params"]
        add_string = ""
        if problem == "cartpole":
            add_string += f"tau: {other_config.get('tau', 0)}"
            add_string += " "
            add_string += f"template: {other_config.get('template', 0)}"
        add_string += f" agent:{other_config['agent']} {other_config['checkpoint']}"
        return f"{problem}_{self.date_time}_{add_string}"


if __name__ == '__main__':
    ray.init(local_mode=False, log_to_driver=False)
    cpu = 7
    trials = list(_iter())
    n_trials = len(trials) - 1
    print(f"Total n of trials: {n_trials}")
    start_from = 0  # 7 stoppign_car
    name_group = NameGroup()
    for i, (problem, method, other_config) in enumerate(trials):
        if i < start_from:
            continue
        print(f"Starting trial: {i}/{n_trials}")
        analysis = tune.run(
            run_parameterised_experiment,
            name="experiment_collection_cartpole_iterations",
            config={
                "main_params": (problem, method, other_config),
                "n_workers": cpu
            },
            resources_per_trial={"cpu": 1},
            stop={"time_since_restore": 500},
            trial_name_creator=name_group.trial_str_creator,
            # resume="PROMPT",
            verbose=0,
            log_to_file=True)
        # df = analysis.results_df
        # df.to_json(os.path.join(analysis.best_logdir, "experiment_results.json"))
        print(f"Finished trial: {i}/{n_trials}")
