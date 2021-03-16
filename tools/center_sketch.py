import os
import json
import numpy as np
import argparse
import random
import math
import copy
from rdp import rdp


def removeSinglePoint(data):
    newData = []
    for stroke in data:
        if len(stroke[0]) > 1:
            newData.append(stroke)
    return newData

def findMinMaxPoint(data):
    """
    data: [[[x0,x1,x2,...],[y0,y1,y2,...]] * storke_number]
    """
    minX = min(list(map(lambda x:min(x[0]), data)))
    maxX = max(list(map(lambda x:max(x[0]), data)))
    minY = min(list(map(lambda x:min(x[1]), data)))
    maxY = max(list(map(lambda x:max(x[1]), data)))
    return minX, minY, maxX, maxY

def centeredSketch(data):
    """
    data: [[[x0,x1,x2,...],[y0,y1,y2,...]] * storke_number]
    """
    def centered(data, scale, old_center, new_center):
        centered_data = []
        for stroke in data:
            x = list(map(lambda x:int((x-old_center[0])*scale + new_center[0]), stroke[0]))
            y = list(map(lambda x:int((x-old_center[1])*scale + new_center[1]), stroke[1]))
            centered_data.append([x, y])
        return centered_data

    def getScale(x0, y0, x1, y1):
        cur_size = max((x1-x0), (y1-y0))
        if cur_size - 254 > 1 or 254 - cur_size > 2:
            cur_scale = 254.0 / cur_size
        else:
            cur_scale = 1
        return cur_scale
    
    x0, y0, x1, y1 = findMinMaxPoint(data)
    scale = getScale(x0, y0, x1, y1)
    centered_sketch_data = centered(data, scale, [(x0+x1)/2, (y0+y1)/2], [127, 127])
    
    while True:
        x0, y0, x1, y1 = findMinMaxPoint(centered_sketch_data)
        if x0 >= 0 and y0 >= 0 and x1 <= 255 and y1 <= 255:
            return centered_sketch_data
        else :
            new_scale = getScale(x0, y0, x1, y1)
            centered_sketch_data = centered(centered_sketch_data, new_scale, [(x0+x1)/2, (y0+y1)/2], [127, 127])


def rotate_theta(data, theta):
    """
    """
    m = np.array([[math.cos(theta), math.sin(theta)],[-math.sin(theta), math.cos(theta)]])
    rotated_data = []
    for stroke in data:
        label = stroke[2]
        stroke_data = np.matmul(m, np.array(stroke[:2])).tolist()
        stroke_data.append(label)
        rotated_data.append(stroke_data)
    return rotated_data

def add_normal_noise(data, scale=1.0):
    noise_data = []
    for stroke in data:
        label = stroke[2]
        stroke_data = np.array(stroke[:2])
        # stroke_data += np.random.normal(0, scale, stroke_data.shape).astype(np.int32)
        stroke_data += (scale*np.random.randn(*stroke_data.shape)).astype(np.int32)
        stroke_data = stroke_data.tolist()
        stroke_data.append(label)
        noise_data.append(stroke_data)
    return noise_data