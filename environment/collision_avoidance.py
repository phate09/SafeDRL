import gym
from gym.utils import seeding
from gym import spaces, logger
import random
import time
import numpy as np
from gym.envs.classic_control import rendering


class ColAvoidEnvDiscrete(gym.Env):
    # metadata = {'render.modes': ['human']}
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, config=None):
        # SIMULATION PARAMETERS
        self.map_size = 20  # length of side (m)
        self.num_dimensions = 2  # number of dimensions
        self.t = 0.20  # time interval (s)

        # VEHICLE PARAMETERS
        self.R = 0.4  # radius (m)
        self.S = 1  # speed (m/s)
        self.num_agents = 1  # number of vehicles
        self.num_directions = 8  # number of moving directions
        self.num_states = 9  # number of states in total
        # (it can also stop, etc)

        # LiDAR PARAMETERS
        self.range_detection = np.array([self.R, 5])
        # detection range (m)
        # (radius of vehicle ~ 5)
        self.range_view = np.array([0, 2 * np.pi])  # field of view
        self.num_bins = 40  # number of bins
        self.resolution = 20  # resolution

        # INTRUDER PARAMETERS
        self.r = 0.4  # radius (m)
        self.s = 2  # speed (m/s)
        self.num_intruders = 1  # number of intruders at the
        # same time

        # STATUS
        # self.EAST, self.NE, self.NORTH, self.NW,
        # self.WEST, self.SW, self.SOUTH, self.SE, self.STOP = range(9)

        self.vel_agents = None  # one of the 9 actions above
        self.pos_agents = None  # position of training
        self.vel_intruders = None  # moving direction of intruders
        self.pos_intruders = None  # position of intruders
        self.observation = None  # 40 bins + 2 dimensions

        # lower bound and upper bound
        min_detection = 0 * np.ones(self.num_bins)  # self.range_detection[0] * np.ones(self.num_bins)
        max_detection = 50 * np.ones(self.num_bins)  # self.range_detection[1] * np.ones(self.num_bins)
        min_location = -self.map_size / 2 * np.ones(self.num_dimensions)
        max_location = self.map_size / 2 * np.ones(self.num_dimensions)

        self.action_space = gym.spaces.Discrete(self.num_states)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(42,),
                                                # np.concatenate((min_detection, min_location)),
                                                # np.concatenate((max_detection, max_location)),
                                                dtype=np.float32)

        self.seed()
        self.viewer = None
        self.steps_beyond_done = None

        self.reward = 0
        self.penalty_1 = 0
        self.penalty_2 = 0
        self.penalty_3 = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        # update the scene
        self.vel_agents = action
        self.update_map()

        # observation
        self.observation = self.observe()

        # done
        done = bool(self.check_collision()
                    or self.check_escape())

        # reward

        # reward 1: survive
        reward_1 = 3.0 if not done else -1000.0

        # penalty 1: distance to (0, 0)
        # 0 ~ 1
        penalty_1 = 0.3 * self.dist_to_position() ** 2  # np.min([self.dist_to_position() ** 2 / 5.0 ** 2, 1.0])

        # penalty_2: distance to closest intruder
        # 0 ~ 1
        penalty_2 = 0  # 1.0 * (self.range_detection[1] - self.dist_to_intruder()) ** 2 / (self.range_detection[1] - self.range_detection[0]) ** 2

        # penalty 3: large changes on speed and direction
        penalty_3 = 0.5 * 1.0 if action != 8 else 0.0

        reward = reward_1 - penalty_1 - penalty_2 - penalty_3
        # if not done:
        #     pass
        #     # reward = reward_1 - 10 * penalty_1 - 1.0 * penalty_2 - 1.0 * penalty_3
        # elif self.steps_beyond_done is None:
        #     # just failed
        #     self.steps_beyond_done = 0
        #     # reward = reward_1 - 10 * penalty_1 - 1.0 * penalty_2 - 1.0 * penalty_3
        # else:
        #     if self.steps_beyond_done == 0:
        #         logger.warn(
        #             "You are calling 'step()' even though this "
        #             "environment has already returned done = True. You "
        #             "should always call 'reset()' once you receive 'done = "
        #             "True' -- any further steps are undefined behavior."
        #         )
        #     self.steps_beyond_done += 1
        #     # reward = reward_1 - 10 * penalty_1 - 1.0 * penalty_2 - 1.0 * penalty_3
        #     self.reset()

        info = {}
        self.reward, self.penalty_1, self.penalty_2, self.penalty_3 = reward, penalty_1, penalty_2, penalty_3
        # print(str(reward) + '\t' + str(penalty_1) + '\t' + str(penalty_2) + '\t' + str(penalty_3))
        return self.observation, reward, done, info

    def reset(self):

        self.vel_agents = np.zeros(self.num_intruders)
        self.pos_agents = np.zeros([self.num_agents, self.num_dimensions])
        self.pos_intruders = None
        self.vel_intruders = None
        self.init_intruders()

        # self.observation = self.range_detection[0] * np.ones(42)
        self.observation = self.observe()

        return self.observation

    def update_map(self, ):
        self.move_agents()
        self.move_intruders()
        self.init_intruders()

    def move_agents(self):
        for i in range(self.num_agents):
            if not self.vel_agents == 9:
                angle = 2 * np.pi / self.num_directions * self.vel_agents
                dx = self.t * self.S * np.cos(angle)
                dy = self.t * self.S * np.sin(angle)
                self.pos_agents[i] = self.pos_agents[i] + np.array([dx, dy])

    def move_intruders(self):
        for i in range(self.num_intruders):
            angle = self.vel_intruders[i]
            dx = self.t * self.s * np.cos(angle)
            dy = self.t * self.s * np.sin(angle)
            self.pos_intruders[i] = self.pos_intruders[i] + np.array([dx, dy])

    def init_intruders(self):
        if self.vel_intruders is None or self.pos_intruders is None:
            self.vel_intruders = np.zeros(self.num_intruders)
            self.pos_intruders = self.map_size * np.ones([self.num_intruders, self.num_dimensions])
            # place them outside the map so that they can be re-initialized

        for i in range(self.num_intruders):
            # check if they lie outside a circle of r > 10
            if np.linalg.norm(self.pos_intruders[i]) > self.map_size / 2:
                # the place to initialize
                # place them on a circle with radius = 10m
                angle = 2 * np.pi * self.np_random.random()
                x = self.map_size / 2 * np.cos(angle)
                y = self.map_size / 2 * np.sin(angle)
                self.pos_intruders[i] = np.array([x, y])

                # # strategy 1: head to a region around (0, 0)
                # # randomly select an angle for each intruder
                # opp_angle = - (np.pi - angle)
                # range_angle = np.arcsin(self.range_detection[1] / (self.map_size / 2))
                # rand_angle = opp_angle - range_angle + (2 * range_angle * self.np_random.random())
                # self.vel_intruders[i] = rand_angle

                # strategy 2: head to the agent directly
                dx = self.pos_agents[0][0] - self.pos_intruders[i][0]
                dy = self.pos_agents[0][1] - self.pos_intruders[i][1]
                self.vel_intruders[i] = np.arctan(dy / dx)

    def observe(self):
        # initialize observation
        if self.observation is None:
            self.observation = np.zeros(self.num_bins + self.num_dimensions)

        for i in range(self.num_bins):
            angle = 2 * np.pi / self.num_bins * i
            for j in range(self.resolution):
                distance = self.range_detection[0] + (self.range_detection[1]
                                                      - self.range_detection[0]) / self.resolution * j
                dx = distance * np.cos(angle)
                dy = distance * np.sin(angle)
                point = self.pos_agents[0] + np.array([dx, dy])

                touch_obstacle = False
                self.observation[i] = distance
                for k in range(self.num_intruders):
                    if np.linalg.norm(point - self.pos_intruders[k]) < self.r:
                        touch_obstacle = True
                        break

                if touch_obstacle:
                    break

        self.observation[-2:] = self.pos_agents[0]
        return self.observation

    def check_collision(self):
        collision = False
        for i in range(self.num_intruders):
            distance = np.linalg.norm(self.pos_agents - self.pos_intruders[i])
            if distance < (self.R + self.r):
                collision = True
                break
        return collision

    def check_escape(self):
        escape = False
        if np.max(np.abs(self.pos_agents[0])) > self.map_size / 2:
            escape = True
        return escape

    def dist_to_position(self):
        distance = np.linalg.norm(self.pos_agents)
        return distance

    def dist_to_intruder(self):
        distance = np.min(self.observation[:-2])
        return distance

    def render(self, mode='human'):
        screen_size = 500
        scale = screen_size / self.map_size

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_size, screen_size)

            self.ranges = [0] * self.num_agents
            self.agents = [0] * self.num_agents
            self.agents_trans = [0] * self.num_agents
            self.intruders = [0] * self.num_intruders
            self.intruders_trans = [0] * self.num_intruders

            for i in range(self.num_agents):
                self.agents_trans[i] = rendering.Transform()

                self.ranges[i] = rendering.make_circle(self.range_detection[1] * scale)
                self.ranges[i].set_color(.9, .9, .6)
                self.ranges[i].add_attr(self.agents_trans[i])
                self.viewer.add_geom(self.ranges[i])

                self.agents[i] = rendering.make_circle(self.R * scale)
                self.agents[i].set_color(.5, .5, .8)
                self.agents[i].add_attr(self.agents_trans[i])
                self.viewer.add_geom(self.agents[i])

            for i in range(self.num_intruders):
                self.intruders[i] = rendering.make_circle(self.r * scale)
                self.intruders_trans[i] = rendering.Transform()
                self.intruders[i].set_color(.8, .5, .5)
                self.intruders[i].add_attr(self.intruders_trans[i])
                self.viewer.add_geom(self.intruders[i])

        if self.observation is None:
            return None

        for i in range(self.num_agents):
            x = self.pos_agents[i][0] * scale + screen_size / 2.0
            y = self.pos_agents[i][1] * scale + screen_size / 2.0
            self.agents_trans[i].set_translation(x, y)

        for i in range(self.num_intruders):
            x = self.pos_intruders[i][0] * scale + screen_size / 2.0
            y = self.pos_intruders[i][1] * scale + screen_size / 2.0
            self.intruders_trans[i].set_translation(x, y)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == '__main__':
    env = ColAvoidEnvDiscrete()
    env.reset()
    env.render()
    for i in range(1000):
        action = env.action_space.sample()
        env.step(action)
        env.render()
    env.close()
