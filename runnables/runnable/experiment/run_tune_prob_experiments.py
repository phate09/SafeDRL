"""
Loads the agents previously trained and then performs the probabilistic verification with various parameters
"""
import datetime
import os
import time

import numpy as np
import ray
from ray import tune

import utils
from polyhedra.experiments_nn_analysis import Experiment
from runnables.runnable.experiment.run_experiment_bouncing_ball import BouncingBallExperiment
from runnables.runnable.experiment.run_experiment_cartpole import CartpoleExperiment
from runnables.runnable.experiment.run_experiment_stopping_car import StoppingCarExperiment
from runnables.runnable.experiment.run_ora_stopping_car import ORAStoppingCarExperiment
from runnables.runnable.experiment.run_prob_experiment_bouncing_ball import BouncingBallExperimentProbabilistic
from runnables.runnable.experiment.run_prob_experiment_pendulum import PendulumExperimentProbabilistic
from runnables.runnable.experiment.run_prob_experiment_stopping_car import StoppingCarExperimentProbabilistic

nn_paths_bouncing_ball = [
    # "tune_PPO_bouncing_ball/PPO_BouncingBall_fb929_00000_0_2021-01-19_00-19-23/checkpoint_10/checkpoint-10",
    # "tune_PPO_bouncing_ball/PPO_BouncingBall_fb929_00001_1_2021-01-19_00-19-23/checkpoint_20/checkpoint-20",
    # "tune_PPO_bouncing_ball/PPO_BouncingBall_fb929_00002_2_2021-01-19_00-20-06/checkpoint_10/checkpoint-10",
    # "tune_PPO_bouncing_ball/PPO_BouncingBall_fb929_00003_3_2021-01-19_00-20-45/checkpoint_10/checkpoint-10",
    # "tune_PPO_bouncing_ball/PPO_BouncingBall_fb929_00004_4_2021-01-19_00-20-52/checkpoint_20/checkpoint-20",
    #
    # "tune_PPO_bouncing_ball/PPO_BouncingBall_c7326_00000_0_2021-01-16_05-43-36/checkpoint_36/checkpoint-36",
    # "tune_PPO_bouncing_ball/PPO_BouncingBall_71684_00000_0_2021-01-18_23-46-54/checkpoint_10/checkpoint-10",
    # "tune_PPO_bouncing_ball/PPO_BouncingBall_71684_00001_1_2021-01-18_23-46-54/checkpoint_10/checkpoint-10",
    # "tune_PPO_bouncing_ball/PPO_BouncingBall_71684_00002_2_2021-01-18_23-47-37/checkpoint_10/checkpoint-10",
    # "tune_PPO_bouncing_ball/PPO_BouncingBall_71684_00003_3_2021-01-18_23-47-37/checkpoint_10/checkpoint-10",
    "tune_PPO_bouncing_ball/PPO_BouncingBall_71684_00004_4_2021-01-18_23-48-21/checkpoint_10/checkpoint-10",
]
nn_paths_stopping_car = [
    "tune_PPO_stopping_car/PPO_StoppingCar_acc24_00001_1_cost_fn=0,epsilon_input=0_2021-01-21_02-30-49/checkpoint_58/checkpoint-58",
    # "tune_PPO_stopping_car/PPO_StoppingCar_14b68_00000_0_cost_fn=0,epsilon_input=0_2021-01-17_11-56-58/checkpoint_31/checkpoint-31",
    # "tune_PPO_stopping_car/PPO_StoppingCar_14b68_00001_1_cost_fn=0,epsilon_input=0.1_2021-01-17_11-56-58/checkpoint_37/checkpoint-37",
    # "tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00000_0_cost_fn=0,epsilon_input=0_2021-01-17_12-37-27/checkpoint_24/checkpoint-24",
    # "tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00001_1_cost_fn=0,epsilon_input=0.1_2021-01-17_12-37-27/checkpoint_36/checkpoint-36",
    # "tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00002_2_cost_fn=0,epsilon_input=0_2021-01-17_12-38-53/checkpoint_40/checkpoint-40",
    # "tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00003_3_cost_fn=0,epsilon_input=0.1_2021-01-17_12-39-31/checkpoint_32/checkpoint-32",
    # "tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00004_4_cost_fn=0,epsilon_input=0_2021-01-17_12-41-14/checkpoint_76/checkpoint-76",
    # "tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00005_5_cost_fn=0,epsilon_input=0.1_2021-01-17_12-41-27/checkpoint_58/checkpoint-58",
    # "tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00006_6_cost_fn=0,epsilon_input=0_2021-01-17_12-44-54/checkpoint_41/checkpoint-41",
    # "tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00007_7_cost_fn=0,epsilon_input=0.1_2021-01-17_12-45-46/checkpoint_89/checkpoint-89",
    # "tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00008_8_cost_fn=0,epsilon_input=0_2021-01-17_12-47-19/checkpoint_43/checkpoint-43",
    # "tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00009_9_cost_fn=0,epsilon_input=0.1_2021-01-17_12-49-48/checkpoint_50/checkpoint-50",
    # "tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00010_10_cost_fn=0,epsilon_input=0_2021-01-17_12-51-01/checkpoint_27/checkpoint-27",
    # "tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00011_11_cost_fn=0,epsilon_input=0.1_2021-01-17_12-52-36/checkpoint_44/checkpoint-44",
    # "tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00012_12_cost_fn=0,epsilon_input=0_2021-01-17_12-52-47/checkpoint_50/checkpoint-50",
    # "tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00013_13_cost_fn=0,epsilon_input=0.1_2021-01-17_12-55-12/checkpoint_53/checkpoint-53",
    # "tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00014_14_cost_fn=0,epsilon_input=0_2021-01-17_12-55-46/checkpoint_56/checkpoint-56",
    # "tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00015_15_cost_fn=0,epsilon_input=0.1_2021-01-17_12-58-23/checkpoint_48/checkpoint-48",
    # "tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00016_16_cost_fn=0,epsilon_input=0_2021-01-17_12-59-01/checkpoint_38/checkpoint-38",
    # "tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00017_17_cost_fn=0,epsilon_input=0.1_2021-01-17_13-01-15/checkpoint_50/checkpoint-50",
    # "tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00018_18_cost_fn=0,epsilon_input=0_2021-01-17_13-01-17/checkpoint_35/checkpoint-35",
    # "tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00019_19_cost_fn=0,epsilon_input=0.1_2021-01-17_13-03-22/checkpoint_36/checkpoint-36"
]
nn_paths_cartpole = ["tune_PPO_pendulum/PPO_MonitoredPendulum_035b5_00000_0_2021-05-11_11-59-52/checkpoint_3333/checkpoint-3333",
                     ]


def _iter():
    for problem in ["bouncing_ball", "stopping_car", "pendulum"]:  # "bouncing_ball", "stopping_car", "pendulum"
        if problem == "bouncing_ball":
            for phi in [0.1]:  # {"tau": tune.grid_search([0.1, 0.05])}
                for initial_state in [0, 1]:
                    for template in [0, 1]:  # box octagon
                        for use_contain in [True, False]:
                            for nn_path in range(0, min(100, len(nn_paths_bouncing_ball))):
                                yield problem, {"phi": phi, "template": template, "initial_state": initial_state, "use_contain": use_contain, "nn_path": nn_path}
        elif problem == "stopping_car":
            for phi in [0.33, 0.5]:  # "epsilon_input": tune.grid_search([0, 0.1])
                for template in [0, 1, 2]:  # box, octagon,template
                    for nn_path in range(0, min(100, len(nn_paths_stopping_car))):
                        yield problem, {"phi": phi, "template": template, "nn_path": nn_path}
        else:
            for phi in [0.5]:  # "tau": tune.grid_search([0.001, 0.02, 0.005]
                for template in [0, 1]:
                    for nn_path in range(0, min(100, len(nn_paths_cartpole))):
                        yield problem, {"phi": phi, "template": template, "nn_path": nn_path}


def update_progress(n_workers, seen, frontier, num_already_visited, max_t):
    tune.report(n_workers=n_workers, seen=seen, frontier=frontier, num_already_visited=num_already_visited, max_t=max_t, safe=0)


def run_parameterised_experiment(config, trial_dir):
    experiment = get_experiment_instance(config, trial_dir=trial_dir)

    stats: Experiment.LoopStats = experiment.run_experiment()

    elapsed_seconds = stats.elapsed_time
    max_elapsed_time = stats.max_elapsed_time
    max_t = stats.max_t
    safe_value = 0
    if stats.is_agent_unsafe is None or elapsed_seconds > max_elapsed_time:
        safe_value = 0
    elif stats.is_agent_unsafe:
        safe_value = -1
    elif not stats.is_agent_unsafe:
        safe_value = 1
    verification_results_path = os.path.join(experiment.save_dir, "verification_result")
    now = datetime.datetime.now()

    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    with open(verification_results_path, "w") as f:
        f.write(f"\nsafe={safe_value}, \nmax_t={max_t}, \ndone={dt_string}, \nelapsed_seconds={elapsed_seconds}, \nmax_elapsed_time={max_elapsed_time}"
                f"\nseen={len(stats.seen)}, \ndiscarded={len(stats.discarded)}, \nfrontier={len(stats.frontier)}, \ngraph_size={experiment.graph.number_of_nodes()}, "
                f"\nconfg={config['main_params']}")
    # tune.report(elapsed_seconds=elapsed_seconds, safe=safe_value, max_t=max_t, done=True)


def get_experiment_instance(config, trial_dir):
    # Hyperparameters
    problem, other_config = config["main_params"]
    n_workers = config["n_workers"]
    if problem == "bouncing_ball":
        experiment = BouncingBallExperimentProbabilistic()
        experiment.nn_path = os.path.join(utils.get_agents_dir(), nn_paths_bouncing_ball[other_config["nn_path"]])
        assert os.path.exists(experiment.nn_path)
        experiment.use_contained = other_config["use_contain"]
        if other_config["initial_state"] == 0:
            experiment.input_boundaries = [9, -5, 0, 0.1]
        elif other_config["initial_state"] == 1:
            experiment.input_boundaries = [9, -5, 1, 1]
        else:
            raise NotImplementedError()
        if other_config["template"] == 1:  # octagon
            experiment.analysis_template = Experiment.octagon(experiment.env_input_size)
        elif other_config["template"] == 0:  # box
            experiment.analysis_template = Experiment.box(experiment.env_input_size)
        else:
            raise NotImplementedError()
        experiment.max_probability_split = other_config["phi"]
        experiment.n_workers = n_workers
        experiment.show_progressbar = False
        experiment.show_progress_plot = False
        experiment.save_dir = trial_dir
        # experiment.update_progress_fn = update_progress
    elif problem == "stopping_car":
        experiment = StoppingCarExperimentProbabilistic()
        experiment.nn_path = os.path.join(utils.get_agents_dir(), nn_paths_stopping_car[other_config["nn_path"]])
        experiment.max_probability_split = other_config["phi"]
        if other_config["template"] == 2:  # octagon
            experiment.analysis_template = Experiment.octagon(experiment.env_input_size)
        elif other_config["template"] == 1:
            delta_x = Experiment.e(experiment.env_input_size, 0)
            v_ego = Experiment.e(experiment.env_input_size, 1)
            template = np.array([delta_x, -delta_x, v_ego, -v_ego, 1 / 4.5 * delta_x + v_ego, 1 / 4.5 * delta_x - v_ego, -1 / 4.5 * delta_x + v_ego,
                                 -1 / 4.5 * delta_x - v_ego])
            experiment.analysis_template = template
        elif other_config["template"] == 0:  # box
            experiment.analysis_template = Experiment.box(experiment.env_input_size)
        else:
            raise NotImplementedError()
        experiment.n_workers = n_workers
        experiment.show_progressbar = False
        experiment.show_progress_plot = False
        experiment.save_dir = trial_dir
        # experiment.update_progress_fn = update_progress
    else:
        experiment = PendulumExperimentProbabilistic()
        experiment.nn_path = os.path.join(utils.get_agents_dir(), nn_paths_cartpole[other_config["nn_path"]])
        if other_config["template"] == 1:  # octagon
            experiment.analysis_template = Experiment.octagon(experiment.env_input_size)
        elif other_config["template"] == 0:  # box
            experiment.analysis_template = Experiment.box(experiment.env_input_size)
        else:
            raise NotImplementedError()
        experiment.max_probability_split = other_config["phi"]
        experiment.n_workers = n_workers
        experiment.show_progressbar = False
        experiment.show_progress_plot = False
        # experiment.use_rounding = False
        experiment.save_dir = trial_dir
        # experiment.update_progress_fn = update_progress
    return experiment


class NameGroup:
    def __init__(self):
        ts = int(time.time())
        d = datetime.datetime.fromtimestamp(ts)
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.date_time = datetime_str
        self.name = hex(int(time.time()))[2:]

    def trial_str_creator(self, config):
        problem, other_config = config["main_params"]
        add_string = ""
        if problem == "cartpole":
            add_string = f"phi: {other_config.get('phi', 0)} template: {other_config.get('template', 0)} agent:{other_config['nn_path']}"
        if problem == "bouncing_ball":
            add_string = f"phi: {other_config.get('phi', 0)} template: {other_config.get('template', 0)} initial_state: {other_config.get('initial_state', 0)} agent:{other_config['nn_path']}"
        if problem == "stopping_car":
            add_string = f"phi: {other_config.get('phi', 0)} template: {other_config.get('template', 0)} agent:{other_config['nn_path']}"
        return os.path.join(self.name, f"{problem} {add_string}")


if __name__ == '__main__':
    ray.init(local_mode=os.environ.get("local_mode", False), log_to_driver=False, include_dashboard=True)
    cpu = 8
    trials = list(_iter())
    n_trials = len(trials) - 1
    print(f"Total n of trials: {n_trials}")
    start_from = 0
    name_group = NameGroup()
    for i, (problem, other_config) in enumerate(trials):
        if i < start_from:
            continue
        print(f"Starting trial: {i + 1}/{n_trials + 1}")
        experiment_config = {
            "main_params": (problem, other_config),
            "n_workers": cpu
        }
        trial_dir = os.path.join(utils.get_save_dir(), "experiment_collection_NFM", name_group.trial_str_creator(experiment_config))
        run_parameterised_experiment(config=experiment_config, trial_dir=trial_dir)
        print(f"Finished trial: {i + 1}/{n_trials + 1}")
