import numpy as np
import math

pi = math.pi

t = 0.05  # 一帧的时间

x = np.array([1, 0, 0])
y = np.array([0, 1, 0])
z = np.array([0, 0, 1])

baseset = {
    'position': np.array([0, 0, 0], dtype=np.float),
    'toward': np.array([math.cos(pi/4), math.sin(pi/4), 0], dtype=np.float),
    'trans_matrix': np.array([[math.cos(pi/4), math.cos(pi*3/4), 0],
                              [math.cos(pi/4), math.cos(pi/4), 0],
                              [0, 0, 1]], dtype=np.float),
    'velocity': 0.,
    'oil': 100.,
    'missile_number': 5
}


def vector_angle(a, b):
    return sum(a * b) / (math.sqrt(sum(a * a)) * math.sqrt(sum(b * b)))


def unitization(a):
    try:
        return a / math.sqrt(sum(a * a))
    except:
        return a


class UCAVs(object):
    def __init__(self, init_states=None):
        if init_states is None:
            init_states = baseset
        self.init_pos = init_states['position']
        self.pos = self.init_pos  # 飞机位置是一个三维数组
        self.init_tow = init_states['toward']
        self.tow = self.init_tow  # 飞机朝向是一个三维数组，表示飞机朝向的方向向量
        self.init_vel = init_states['velocity']
        self.vel = 0.
        self.init_oil = init_states['oil']
        self.oil = 0.
        self.init_mis_num = init_states['missile_number']
        self.mis_num = 0
        self.mis = []
        self.init_trans_matrix = init_states['trans_matrix']
        self.trans_matrix = self.init_trans_matrix
        self.alive = True
        self.sight1 = 300.
        self.sight2 = 200.

        self.reset()

    def reset(self):
        self.pos = self.init_pos
        self.tow = self.init_tow
        self.vel = self.init_vel
        self.oil = self.init_oil
        self.mis_num = self.init_mis_num
        self.trans_matrix = self.init_trans_matrix
        self.mis = []

    def launch(self, delta):
        if delta == 1:
            self.mis_num -= 1
            self.mis.append(missile(self.pos, self.tow, self.vel))

    def pitch(self, delta):
        x0 = np.array([1, 0, math.tan(delta)])
        z0 = np.array([-math.tan(delta), 0, 1])
        x1 = np.dot(self.trans_matrix, x0)
        z1 = np.dot(self.trans_matrix, z0)
        self.trans_matrix[0][0] = vector_angle(x, x1)
        self.trans_matrix[0][2] = vector_angle(x, z1)
        self.trans_matrix[1][0] = vector_angle(y, x1)
        self.trans_matrix[1][2] = vector_angle(y, z1)
        self.trans_matrix[2][0] = vector_angle(z, x1)
        self.trans_matrix[2][2] = vector_angle(z, z1)
        self.tow = unitization(x1)

    def roll(self, delta):
        y0 = np.array([0, 1, math.tan(delta)])
        z0 = np.array([0, -math.tan(delta), 1])
        y1 = np.dot(self.trans_matrix, y0)
        z1 = np.dot(self.trans_matrix, z0)
        self.trans_matrix[0][1] = vector_angle(x, y1)
        self.trans_matrix[0][2] = vector_angle(x, z1)
        self.trans_matrix[1][1] = vector_angle(y, y1)
        self.trans_matrix[1][2] = vector_angle(y, z1)
        self.trans_matrix[2][1] = vector_angle(z, y1)
        self.trans_matrix[2][2] = vector_angle(z, z1)

    def yaw(self, delta):
        pass

    def velocity_change(self, delta):
        self.vel += delta

    def oil_change(self):
        self.oil -= self.vel * 0.01

    def move(self):
        self.pos = self.pos + self.tow * self.vel * t


class missile(object):
    def __init__(self, position, toward, velocity):
        self.pos = position
        self.tow = toward
        self.vel = velocity
        self.steps = 0
        self.delta = 1/(2*pi)
        self.blast_range = 1.

    def update(self, direction=None):
        if direction is None:
            direction = self.tow
        else:
            direction = unitization(direction)
        if vector_angle(self.tow, direction) > math.cos(self.delta):
            self.tow = direction
        else:
            theta = math.acos(vector_angle(self.tow, direction))
            x0 = np.array([math.cos(self.delta)-math.sin(self.delta)/math.tan(theta),
                           math.sin(self.delta)/math.sin(theta)], dtype=np.float)
            trans = np.array([[vector_angle(self.tow, x), vector_angle(direction, x)],
                              [vector_angle(self.tow, y), vector_angle(direction, y)],
                              [vector_angle(self.tow, z), vector_angle(direction, z)]], dtype=np.float)
            x1 = np.dot(trans, x0)
            self.tow = unitization(x1)
        self.vel_update()
        self.pos = self.pos + self.tow * self.vel * t
        self.steps += 1

    # 速度变化曲线，由self.steps和t决定
    def vel_update(self):
        pass
