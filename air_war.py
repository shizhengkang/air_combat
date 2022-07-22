import math
import gym
from gym import spaces, logger
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np

class AirWarEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self):
        self.grid_size = 150
        self.fighter_number = 2
        self.action_space = spaces.Tuple((spaces.MultiDiscrete([6, 6]), spaces.MultiDiscrete([6, 6])))
        self.observation_space = spaces.Tuple([
            spaces.Box(
                low=0, high=self.grid_size, shape=(2, 54), dtype=np.int32),
            spaces.Box(
                low=0, high=self.grid_size, shape=(2, 54), dtype=np.int32)
        ])  # 4架飞机的位置坐标，20枚导弹的位置坐标，每架飞机剩余导弹数量，场上剩余敌机数量与我方飞机数量，4*2+20*2+4+2=54
        self.fighter_speed = 1
        self.missile_speed = 2
        self.sight = 300
        self.fighter_remain = [self.fighter_number] * 2
        self.missile_local_remain = [5] * self.fighter_number
        self.missile_enemy_remain = [5] * self.fighter_number
        self.fighter_local_x = [self.grid_size / 3, self.grid_size / 3 + 1]
        self.fighter_local_y = [self.grid_size / 3, self.grid_size / 3]
        self.fighter_enemy_x = [self.grid_size * 2 / 3, self.grid_size * 2 / 3-1]
        self.fighter_enemy_y = [self.grid_size * 2 / 3, self.grid_size * 2 / 3]
        # 导弹初始坐标设置为（-1，-1），其中，坐标（-1，-1）表示导弹未发射，坐标（-2，-2）表示销毁导弹（打中或未打中），坐标（-3，-3）表示不在视野范围内的导弹
        self.missile_local_x = [-1] * self.fighter_number * 5
        self.missile_local_y = [-1] * self.fighter_number * 5
        self.missile_enemy_x = [-1] * self.fighter_number * 5
        self.missile_enemy_y = [-1] * self.fighter_number * 5
        self.missile_local_distance = [0] * self.fighter_number * 5
        self.missile_enemy_distance = [0] * self.fighter_number * 5

        self.state = np.array([self.fighter_local_x, self.fighter_local_y, self.fighter_enemy_x, self.fighter_enemy_y,
                      self.missile_local_x, self.missile_local_y, self.missile_enemy_x, self.missile_enemy_y,
                      self.fighter_remain, self.missile_local_remain, self.missile_enemy_remain,
                      self.missile_local_distance, self.missile_enemy_distance])

        self.viewer = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
         state: eg.

         action: eg. [[1,2],[2,3]]；
         action[0]表示本方动作，action[1]表示敌方动作；
         其中1，2，3，4分别表示上，下，左，右四个动作；
         0表示静止无动作，5表示发射导弹。
        """
        # reward的设置规则还需要斟酌，每方的每架战机是否分开给reward？

        flx, fly, fex, fey, mlx, mly, mex, mey, fr, mlr, mer, mld, med = self.state
        reward = [0, 0]
        result, done = 0, 0  # result 表示对局结果，0表示对局未结束，1，2，3分别表示赢、输、平
        self.track()  # 先更新导弹轨迹
        self.is_hit()  # 判断是否有飞机被击中
        for i in range(self.fighter_number):
            if flx[i] >= 0:
                distance1 = 600
                for j in range(self.fighter_number * 5):
                    if mex[j] >= 0 and self.distance(flx[i], mex[j], fly[i], mey[j]) < distance1:
                        distance1 = self.distance(flx[i], mex[j], fly[i], mey[j])
                if action[0][i] == 0:
                    if fly[i] >= self.grid_size:
                        reward[0] -= 2
                    else:
                        fly[i] += 1
                elif action[0][i] == 1:
                    if fly[i] <= 0:
                        reward[0] -= 2
                    else:
                        fly[i] -= 1
                elif action[0][i] == 2:
                    if flx[i] <= 0:
                        reward[0] -= 2
                    else:
                        flx[i] -= 1
                elif action[0][i] == 3:
                    if flx[i] >= self.grid_size:
                        reward[0] -= 2
                    else:
                        flx[i] += 1
                elif action[0][i] == 4:
                    if mlr[i] == 0:
                        reward[0] -= 2
                    else:
                        min_distance = 600
                        for j in range(self.fighter_number):
                            if fex[j] >= 0 and self.distance(fex[j], flx[i], fey[j], fly[i]) < min_distance:
                                min_distance = self.distance(fex[j], flx[i], fey[j], fly[i])
                        for j in range(5):
                            if mlx[i * 5 + j] == -1:
                                mlr[i] -= 1
                                mlx[i * 5 + j], mly[i * 5 + j] = flx[i], fly[i]
                                reward[0] = reward[0] + self.probability(min_distance) - 0.5
                                break
                distance2 = 600
                for j in range(self.fighter_number * 5):
                    if mex[j] >= 0 and self.distance(flx[i], mex[j], fly[i], mey[j]) < distance2:
                        distance2 = self.distance(flx[i], mex[j], fly[i], mey[j])
                if distance1 > distance2:
                    reward[0] -= 0.05
                elif distance1 < distance2:
                    reward[0] += 0.05
        for i in range(self.fighter_number):
            if fex[i] >= 0:
                if action[1][i] == 0:
                    fey[i] = min(fey[i] + 1, self.grid_size)
                elif action[1][i] == 1:
                    fey[i] = max(fey[i] - 1, 0)
                elif action[1][i] == 2:
                    fex[i] = max(fex[i] - 1, 0)
                elif action[1][i] == 3:
                    fex[i] = min(fex[i] + 1, self.grid_size)
                elif action[1][i] == 4:
                    for j in range(5):
                        if mex[i * 5 + j] == -1:
                            mer[i] -= 1
                            mex[i * 5 + j], mey[i * 5 + j] = fex[i], fey[i]
                            reward[1] -= 2
                            break

        # min_distance = 600
        # for i in range(self.fighter_number):
        #     for j in range(self.fighter_number * 5):
        #         if mex[j] >= 0 and self.distance(flx[i], mex[j], fly[i], mey[j]) < min_distance:
        #             min_distance = self.distance(flx[i], mex[j], fly[i], mey[j])
        # reward[0] = reward[0] - self.probability(min_distance) * 20 + 1

        if fr[0] != 0 and fr[1] == 0:
            result, done = 1, 1
        elif fr[0] == 0 and fr[1] != 0:
            result, done = 2, 1
        elif fr[0] == 0 and fr[1] == 0:
            result, done = 3, 1
        elif sum(mlx) == -20 and sum(mex) == -20:
            if fr[0] > fr[1]:
                result, done = 1, 1
            elif fr[0] < fr[1]:
                result, done = 2, 1
            else:
                result, done = 3, 1

        # reward[0] = 0
        if result == 1:
            reward[0] += 10
            reward[1] -= 10
        elif result == 2:
            reward[0] -= 10
            reward[1] += 10

        return self.state, reward[0], done, result

    def reset(self):
        self.fighter_remain = [self.fighter_number] * 2
        self.missile_local_remain = [5] * self.fighter_number
        self.missile_enemy_remain = [5] * self.fighter_number
        # self.fighter_local_x = [np.random.uniform(0, self.grid_size), np.random.uniform(0, self.grid_size)]
        # self.fighter_local_y = [np.random.uniform(0, self.grid_size), np.random.uniform(0, self.grid_size)]
        # self.fighter_enemy_x = [np.random.uniform(0, self.grid_size), np.random.uniform(0, self.grid_size)]
        # self.fighter_enemy_y = [np.random.uniform(0, self.grid_size), np.random.uniform(0, self.grid_size)]
        self.fighter_local_x = [self.grid_size / 3, self.grid_size / 3 + 1]
        self.fighter_local_y = [self.grid_size / 3, self.grid_size / 3]
        self.fighter_enemy_x = [self.grid_size * 2 / 3, self.grid_size * 2 / 3 - 1]
        self.fighter_enemy_y = [self.grid_size * 2 / 3, self.grid_size * 2 / 3]
        # self.fighter_enemy_x = [self.grid_size / 3 + 10, self.grid_size / 3 + 11]
        # self.fighter_enemy_y = [self.grid_size / 3 + 10, self.grid_size / 3 + 11]
        # 导弹初始坐标设置为（-1，-1），其中，坐标（-1，-1）表示导弹未发射，坐标（-2，-2）表示销毁导弹（打中或未打中），坐标（-3，-3）表示不在视野范围内的导弹
        self.missile_local_x = [-1] * self.fighter_number * 5
        self.missile_local_y = [-1] * self.fighter_number * 5
        self.missile_enemy_x = [-1] * self.fighter_number * 5
        self.missile_enemy_y = [-1] * self.fighter_number * 5
        self.missile_local_distance = [0] * self.fighter_number * 5
        self.missile_enemy_distance = [0] * self.fighter_number * 5
        self.state = np.array([self.fighter_local_x, self.fighter_local_y, self.fighter_enemy_x, self.fighter_enemy_y,
                      self.missile_local_x, self.missile_local_y, self.missile_enemy_x, self.missile_enemy_y,
                      self.fighter_remain, self.missile_local_remain, self.missile_enemy_remain,
                      self.missile_local_distance, self.missile_enemy_distance])
        return self.state

    def render(self, mode='human'):
        flx, fly, fex, fey, mlx, mly, mex, mey, fr, mlr, mer, mld, med = self.state
        if self.viewer is None:
            self.viewer = rendering.Viewer((self.grid_size + 2) * 3, (self.grid_size + 2) * 3)

        # 渲染飞机
        for i in range(self.fighter_number):
            if flx[i] >= 0:
                l, m, r, t, b = 3 * flx[i] - 6, 3 * flx[i], 3 * flx[i] + 6, 3 * fly[i] + 6, 3 * fly[i] - 6
                fighter = rendering.FilledPolygon([(l, b), (r, b), (m, t)])
                fighter.set_color(0, 0, 1)
                self.viewer.add_onetime(fighter)
            if fex[i] >= 0:
                l, m, r, t, b = 3 * fex[i] - 6, 3 * fex[i], 3 * fex[i] + 6, 3 * fey[i] + 6, 3 * fey[i] - 6
                fighter = rendering.FilledPolygon([(l, b), (r, b), (m, t)])
                fighter.set_color(1, 0, 0)
                self.viewer.add_onetime(fighter)

        # 渲染导弹
        for i in range(self.fighter_number * 5):
            if mlx[i] >= 0:
                x, y = mlx[i], mly[i]
                missile = rendering.make_circle(2)
                missile_transform = rendering.Transform(translation=(3 * x, 3 * y))
                missile.add_attr(missile_transform)
                self.viewer.add_onetime(missile)
            if mex[i] >= 0:
                x, y = mex[i], mey[i]
                missile = rendering.make_circle(2)
                missile_transform = rendering.Transform(translation=(3 * x, 3 * y))
                missile.add_attr(missile_transform)
                self.viewer.add_onetime(missile)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


    # 更新已发射导弹的轨迹
    def track(self):
        flx, fly, fex, fey, mlx, mly, mex, mey, fr, mlr, mer, mld, med = self.state
        for i in range(self.fighter_number * 5):
            if mlx[i] >= 0:
                mld[i] += 2
                min_distance, min_index = 600, -1
                for j in range(self.fighter_number):
                    if fex[j] >= 0 and self.distance(fex[j], mlx[i], fey[j], mly[i]) < min_distance:
                        min_distance, min_index = self.distance(fex[j], mlx[i], fey[j], mly[i]), j
                if min_distance <= 2:
                    mlx[i], mly[i] = fex[min_index], fey[min_index]
                elif mlx[i] == fex[min_index]:
                    mly[i] = mly[i] + 2 * np.int((fey[min_index] - mly[i]) / abs(fey[min_index] - mly[i]))
                elif mly[i] == fey[min_index]:
                    mlx[i] = mlx[i] + 2 * np.int((fex[min_index] - mlx[i]) / abs(fex[min_index] - mlx[i]))
                else:
                    mlx[i] = mlx[i] + np.int((fex[min_index] - mlx[i]) / abs(fex[min_index] - mlx[i]))
                    mly[i] = mly[i] + np.int((fey[min_index] - mly[i]) / abs(fey[min_index] - mly[i]))

        for i in range(self.fighter_number * 5):
            if mex[i] >= 0:
                med[i] += 2
                min_distance, min_index = 600, -1
                for j in range(self.fighter_number):
                    if flx[j] >= 0 and self.distance(flx[j], mex[i], fly[j], mey[i]) < min_distance:
                        min_distance, min_index = self.distance(flx[j], mex[i], fly[j], mey[i]), j
                if min_distance <= 2:
                    mex[i], mey[i] = flx[min_index], fly[min_index]
                elif mex[i] == flx[min_index]:
                    mey[i] = mey[i] + 2 * np.int((fly[min_index] - mey[i]) / abs(fly[min_index] - mey[i]))
                elif mey[i] == fly[min_index]:
                    mex[i] = mex[i] + 2 * np.int((flx[min_index] - mex[i]) / abs(flx[min_index] - mex[i]))
                else:
                    mex[i] = mex[i] + np.int((flx[min_index] - mex[i]) / abs(flx[min_index] - mex[i]))
                    mey[i] = mey[i] + np.int((fly[min_index] - mey[i]) / abs(fly[min_index] - mey[i]))

        return self.state

    def distance(self, x1, x2, y1, y2):
        return abs(x1 - x2) + abs(y1 - y2)

    def is_hit(self):
        flx, fly, fex, fey, mlx, mly, mex, mey, fr, mlr, mer, mld, med = self.state
        for i in range(self.fighter_number):
            for j in range(self.fighter_number * 5):
                if fex[i] >= 0 and mlx[j] == fex[i] and mly[j] == fey[i]:
                    mlx[j], mly[j] = -2, -2
                    P = self.probability(mld[j])
                    if np.random.choice([0, 1], 1, False, [1-P, P]) == 1:
                        fex[i], fey[i] = -1, -1
                        fr[1] -= 1
                        for k in range(5):
                            if mex[i * 5 + k] == -1:
                                mex[i * 5 + k], mey[i * 5 + k] = -2, -2
                                mer[i] -= 1

        for i in range(self.fighter_number):
            for j in range(self.fighter_number * 5):
                if flx[i] >= 0 and mex[j] == flx[i] and mey[j] == fly[i]:
                    mex[j], mey[j] = -2, -2
                    P = self.probability(med[j])
                    if np.random.choice([0, 1], 1, False, [1-P, P]) == 1:
                        flx[i], fly[i] = -1, -1
                        fr[0] -= 1
                        for k in range(5):
                            if mlx[i * 5 + k] == -1:
                                mlx[i * 5 + k], mly[i * 5 + k] = -2, -2
                                mlr[i] -= 1
        return self.state

    def probability(self, x):
        if x <= 10:
            return x * 0.1
        else:
            return 10 / x

    # 获取每个智能体所能观测到的信息，其中obs为与环境交互获得的全部状态信息，lore取值0/1分别表示友机与敌机，id为具体第几架飞机
    # 由于每个智能体的动作网络是独立的，暂时不把智能体本身的状态信息分离出来，后续可考虑改进
    # 目前设定视野范围内的敌方导弹只有自己知道无法共享给队友，明确后确定是否更改
    def get_actor_observation(self, obs, lore, id):
        FLX, FLY, FEX, FEY, MLX, MLY, MEX, MEY, FR, MLR, MER, MLD, MED = np.array(obs).tolist()
        flx, fly, fex, fey, mlx, mly, mex, mey, fr, mlr, mer, mld, med = \
            FLX[:], FLY[:], FEX[:], FEY[:], MLX[:], MLY[:], MEX[:], MEY[:], FR[:], MLR[:], MER[:], MLD[:], MED[:]
        if lore == 0:
            mes = [0]
            for i in range(self.fighter_number * 5):
                if mex[i] >= 0 and self.distance(flx[id], mex[i], fly[id], mey[i]) > self.sight or mex[i] == -1:
                    mex[i], mey[i] = -3, -3
                elif mex[i] == -2:
                    mes[0] += 1
            x, y = flx[id], fly[id]
            mx, my = [-1], [-1]
            min_distance = 600
            for i in range(self.fighter_number * 5):
                if mex[i] >= 0 and self.distance(flx[id], mex[i], fly[id], mey[i]) < min_distance:
                    min_distance = self.distance(flx[id], mex[i], fly[id], mey[i])
                    mx[0], my[0] = mex[i], mey[i]
            del flx[id]
            del fly[id]
            coordinate_x = flx + fex + mx
            for i in range(len(coordinate_x)):
                if coordinate_x[i] < 0:
                    coordinate_x[i] = x
            coordinate_x = np.array(coordinate_x, dtype=np.float)
            coordinate_x -= x
            coordinate_y = fly + fey + my
            for i in range(len(coordinate_y)):
                if coordinate_y[i] < 0:
                    coordinate_y[i] = y
            coordinate_y = np.array(coordinate_y, dtype=np.float)
            coordinate_y -= y
            location = [x, y]
            state = coordinate_x.tolist() + coordinate_y.tolist() + fr + [mlr[id]] + mes + location

        else:
            mls = [0]
            for i in range(self.fighter_number * 5):
                if mlx[i] >= 0 and self.distance(fex[id], mlx[i], fey[id], mly[i]) > self.sight or mlx[i] == -1:
                    mlx[i], mly[i] = -3, -3
                elif mlx[i] == -2:
                    mls[0] += 1
            x, y = fex[id], fey[id]
            mx, my = [-1], [-1]
            min_distance = 600
            for i in range(self.fighter_number * 5):
                if mlx[i] >= 0 and self.distance(fex[id], mlx[i], fey[id], mly[i]) < min_distance:
                    min_distance = self.distance(fex[id], mlx[i], fey[id], mly[i])
                    mx[0], my[0] = mlx[i], mly[i]
            del fex[id]
            del fey[id]
            coordinate_x = fex + flx + mx
            for i in range(len(coordinate_x)):
                if coordinate_x[i] < 0:
                    coordinate_x[i] = x
            coordinate_x = np.array(coordinate_x, dtype=np.float)
            coordinate_x -= x
            coordinate_y = fey + fly + my
            for i in range(len(coordinate_y)):
                if coordinate_y[i] < 0:
                    coordinate_y[i] = y
            coordinate_y = np.array(coordinate_y, dtype=np.float)
            coordinate_y -= y
            location = [x, y]
            state = coordinate_x.tolist() + coordinate_y.tolist() + fr[::-1] + [mer[id]] + mls + location
        states = np.array(state, dtype=np.float)
        states[0:8] /= 150
        states[8:10] /= 2
        states[10] /= 5
        states[11] /= 10
        states[12:14] /= 150
        return states.tolist()

    # 虽然博弈是不完全信息的博弈，但是在训练时，critic网络是否可以直接利用全局信息！
    def get_critic_observation(self, obs, lore):
        # gobs = self.get_actor_observation(obs, lore, 0)
        # for i in range(self.fighter_number - 1):
        #     gobs = np.maximum(self.get_actor_observation(obs, lore, i), gobs)
        # return gobs
        FLX, FLY, FEX, FEY, MLX, MLY, MEX, MEY, FR, MLR, MER, MLD, MED = np.array(obs).tolist()
        flx, fly, fex, fey, mlx, mly, mex, mey, fr, mlr, mer, mld, med = \
            FLX[:], FLY[:], FEX[:], FEY[:], MLX[:], MLY[:], MEX[:], MEY[:], FR[:], MLR[:], MER[:], MLD[:], MED[:]
        if lore == 0:
            state = flx + fly + fex + fey + mlx + mly + mex + mey + fr + mlr + mer
        else:
            state = fex + fey + flx + fly + mex + mey + mlx + mly + fr[::-1] + mer + mlr
        states = np.array(state, dtype=np.float)
        states[0:48] /= 150
        states[48:50] /= 2
        states[50:54] /= 5
        return states.tolist()

    def get_avail_agent_actions(self, obs):
        avail_actions = np.ones(5)
        if obs[10] == 0:
            avail_actions[4] = 0
        if obs[12] == 0:
            avail_actions[2] = 0
        elif obs[12] == 1:
            avail_actions[3] = 0
        if obs[13] == 0:
            avail_actions[1] = 0
        elif obs[13] == 1:
            avail_actions[0] = 0
        return avail_actions

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
