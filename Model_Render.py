import math
import numpy as np


def unitization(a):
    try:
        return a / math.sqrt(sum(a * a))
    except:
        return a


def flight_model(pos, tow, nor):
    tow = unitization(tow)
    nor = unitization(nor)
    body = []

    body.append(pos + 1.5 * tow)
    body.append(pos + 0.5 * nor)
    body.append(pos - 2 * tow)
    body.append(pos - 0.5 * nor)
    body.append(pos + 1.5 * tow)
    y = np.cross(nor, tow)
    body.append(pos + 1 * y)
    body.append(pos - 2 * tow)
    body.append(pos - 1 * y)
    body.append(pos + 1.5 * tow)

    body.append(pos + 0.3 * tow + 0.8 * y)
    body.append(pos + 2.5 * y)
    body.append(pos + 2.5 * y - 0.5 * tow)
    body.append(pos - tow + 0.5 * y)

    body.append(pos - 1.6 * tow + 0.2 * y)
    body.append(pos - 1.8 * tow + 1.2 * y)
    body.append(pos - 2.2 * tow + 1 * y)
    body.append(pos - 2 * tow)

    body.append(pos - 2.2 * tow - 1 * y)
    body.append(pos - 1.8 * tow - 1.2 * y)
    body.append(pos - 1.6 * tow - 0.2 * y)

    body.append(pos - tow - 0.5 * y)
    body.append(pos - 2.5 * y - 0.5 * tow)
    body.append(pos - 2.5 * y)
    body.append(pos + 0.3 * tow - 0.8 * y)
    body = np.array(body)
    return body.T


def missile_model(pos, tow):
    tow = unitization(tow)
    body = []

    body.append(pos + 0.5 * tow)
    body.append(pos - 0.5 * tow)
    body = np.array(body)
    return body.T
