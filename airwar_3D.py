import math
import gym
from gym import spaces, logger
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np
from Agent import UCAVs
import math
import pygame
import matplotlib.pyplot as plt
from Model_Render import flight_model, missile_model


def distance(x1, x2):
    return math.sqrt(sum((x1 - x2) * (x1 - x2)))


def unitization(a):
    try:
        return a / math.sqrt(sum(a * a))
    except:
        return a


class AirWarEnv3D(gym.Env):
    def __init__(self):
        self.grid_size = 50
        self.agent_number = 1
        self.agent_ctrl = []
        self.agent_oppo = []
        for i in range(self.agent_number):
            self.agent_ctrl.append(UCAVs())
            self.agent_oppo.append(UCAVs())
        self.agents = [self.agent_ctrl, self.agent_oppo]
        self.agent_remain = [self.agent_number] * 2
        self.state = self.get_state()
        self.viewer = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        action是一个2*self.agent_number*4维输入列表，其中：
        action[0]与action[1]分别为博弈双方的动作，
        action[:][j]表示某个阵营第j个智能体的动作，
        action[:][:][0]表示速度的增减，取值范围为[]
        action[:][:][1]表示俯仰角的改变，取值范围为[]
        action[:][:][2]表示滚转角的改变，取值范围为[]
        action[:][:][3]表示是否发射炮弹，取值范围为{0, 1}
        """

        for i in range(2):
            for j in range(self.agent_number):
                if self.agents[i][j].alive:
                    self.agents[i][j].velocity_change(action[i][j][0])
                    self.agents[i][j].pitch(action[i][j][1])
                    self.agents[i][j].roll(action[i][j][2])
                    self.agents[i][j].move()
                    self.agents[i][j].oil_change()
                    self.agents[i][j].launch(action[i][j][3])
                k = 0
                while k < len(self.agents[i][j].mis):
                    self.track(i, j, k)
                    k += 1

        result, done = self.is_terminal()

        reward = 0.
        if result == 1:
            reward += 10.
        elif result == 2:
            reward -= 10.

        self.state = self.get_state()
        return self.state, reward, done, result

    def reset(self):
        for i in range(self.agent_number):
            self.agent_ctrl[i].reset()
            self.agent_oppo[i].reset()
        self.state = self.get_state()
        return self.state

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = plt.axes(projection='3d')
            self.viewer.set_xlim((0, self.grid_size))
            self.viewer.set_ylim((0, self.grid_size))
            self.viewer.set_zlim((0, self.grid_size))
            plt.ion()

        objects = []
        color = ['b', 'r']
        for i in range(2):
            for j in range(self.agent_number):
                if self.agents[i][j].alive:
                    point = flight_model(self.agents[i][j].pos, self.agents[i][j].tow, self.agents[i][j].nor)
                    flight, = self.viewer.plot3D(point[0], point[1], point[2], c=color[i], linewidth=0.5)
                    objects.append(flight)
                for k in range(len(self.agents[i][j].mis)):
                    point = missile_model(self.agents[i][j].mis[k].pos, self.agents[i][j].mis[k].tow)
                    missile, = self.viewer.plot3D(point[0], point[1], point[2], c='black', linewidth=0.5)
                    objects.append(missile)
        plt.show()
        plt.pause(0.01)
        for object_ in objects:
            object_.remove()

    def track(self, i, j, k):
        index = None
        min_distance = float('inf')
        direction = None
        for n in range(self.agent_number):
            if self.agents[1-i][n].alive:
                direc = self.agents[1-i][n].pos - self.agents[i][j].mis[k].pos
                distance = sum(direc * direc)
                if distance < min_distance:
                    min_distance = distance
                    direction = direc
                    index = n
        if min_distance < self.agents[i][j].mis[k].blast_range:
            self.agents[1-i][index].alive = 0
            del self.agents[i][j].mis[k]
        else:
            self.agents[i][j].mis[k].update(direction)

    def is_terminal(self):
        if self.agent_remain[0] != 0 and self.agent_remain[1] == 0:
            return 1, 1
        elif self.agent_remain[0] == 0 and self.agent_remain[1] != 0:
            return 2, 1
        elif self.agent_remain[0] == 0 and self.agent_remain[1] == 0:
            return 3, 1
        elif sum(self.agent_ctrl[i].mis_num * self.agent_ctrl[i].alive for i in range(self.agent_number)) == 0 and \
                sum(self.agent_oppo[i].mis_num * self.agent_oppo[i].alive for i in range(self.agent_number)) == 0:
            if self.agent_remain[0] > self.agent_remain[1]:
                return 1, 1
            elif self.agent_remain[0] < self.agent_remain[1]:
                return 2, 1
            else:
                return 3, 1
        return 0, 0

    def get_obs(self):
        obs = []
        for i in range(2):
            ob = []
            for j in range(self.agent_number):
                if self.agents[i][j].alive:
                    # 自己的信息
                    o = []
                    o += (self.agents[i][j].pos / self.grid_size).tolist()
                    o += self.agents[i][j].tow.tolist()
                    o += self.agents[i][j].nor.tolist()
                    o += [self.agents[i][j].vel / 100.]
                    o += [self.agents[i][j].oil / 100.]
                    o += [self.agents[i][j].mis_num / self.agent_ctrl[j].init_mis_num]
                    for k in range(self.agents[i][j].init_mis_num):
                        try:
                            o += (self.agents[i][j].mis[k].pos / self.grid_size).tolist()
                            o += self.agents[i][j].mis[k].tow.tolist()
                            o += [self.agents[i][j].mis[k].vel / 200.]
                            o += [self.agents[i][j].mis[k].steps / 300]
                        except:
                            o += [0.] * 8
                    # 队友的信息
                    for k in range(self.agent_number):
                        if k != j:
                            o += (self.agents[i][k].pos / self.grid_size).tolist()
                            o += self.agents[i][k].tow.tolist()
                            o += self.agents[i][k].nor.tolist()
                            o += [self.agents[i][k].vel / 100.]
                            o += [self.agents[i][k].oil / 100.]
                            o += [self.agents[i][k].mis_num / self.agent_ctrl[k].init_mis_num]
                            for n in range(self.agents[i][k].init_mis_num):
                                try:
                                    o += (self.agents[i][k].mis[n].pos / self.grid_size).tolist()
                                    o += self.agents[i][k].mis[n].tow.tolist()
                                    o += [self.agents[i][k].mis[n].vel / 200.]
                                    o += [self.agents[i][k].mis[n].steps / 300]
                                except:
                                    o += [0.] * 8
                    # 对手的信息
                    for k in range(self.agent_number):
                        if self.agents[1-i][k].alive and distance(self.agents[i][j].pos, self.agents[1-i][k].pos) < self.agents[i][j].sight1:
                            o += ((self.agents[1-i][k].pos - self.agents[i][j].pos) / self.grid_size).tolist()
                        else:
                            o += [0.] * 3
                        for n in range(self.agents[i][j].init_mis_num):
                            try:
                                if distance(self.agents[i][j].pos, self.agents[1-i][k].mis[n].pos) < self.agents[i][j].sight2:
                                    o += ((self.agents[1-i][k].mis[n].pos - self.agents[i][j].pos) / self.grid_size).tolist()
                                else:
                                    o += [0.] * 3
                            except:
                                o += [0.] * 3
                else:
                    o = [0.] * (12 * self.agent_number + 11 * self.agent_number * self.agents[i][j].init_mis_num)
                ob.append(o)
            obs.append(ob)
        return obs

    def get_state(self):
        states = []
        for i in range(2):
            for j in range(self.agent_number):
                if self.agents[i][j].alive:
                    states += (self.agents[i][j].pos / self.grid_size).tolist()
                    states += self.agents[i][j].tow.tolist()
                    states += self.agents[i][j].nor.tolist()
                    states += [self.agents[i][j].vel / 100.]
                    states += [self.agents[i][j].oil / 100.]
                    states += [self.agents[i][j].mis_num / self.agent_ctrl[j].init_mis_num]
                else:
                    states += [0.] * 9
                for k in range(self.agents[i][j].init_mis_num):
                    try:
                        states += (self.agents[i][j].mis[k].pos / self.grid_size).tolist()
                        states += self.agents[i][j].mis[k].tow.tolist()
                        states += [self.agents[i][j].mis[k].vel / 200.]
                        states += [self.agents[i][j].mis[k].steps / 300]
                    except:
                        states += [0.] * 8
        return states

    def get_avail_agent_actions(self):
        pass

    def close(self):
        if self.viewer:
            plt.close()
            self.viewer = None
