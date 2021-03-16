import os
import ndjson
import cv2 as cv
import numpy as np
from rdp import rdp
from draw_sketch import drawColor

def cal_point_number(sketch):
    point_num = sum(list(map(lambda x:len(x[0]), sketch)))
    return point_num

def pldist(point, start, end):
    if np.all(np.equal(start, end)):
        return np.linalg.norm(point - start)

    return np.divide(
            np.abs(np.linalg.norm(np.cross(end - start, start - point))),
            np.linalg.norm(end - start))

def rdp_iter(M, stk, epsilon, dist=pldist):
    point_num, _ = M.shape
    dists = np.ones(point_num) * 1000

    while stk:
        start_index, last_index = stk.pop()

        dmax = 0.0
        index = start_index

        for i in range(index + 1, last_index):
            d = dist(M[i], M[start_index], M[last_index])
            dists[i] = d
            if d > dmax:
                index = i
                dmax = d

        if dmax > epsilon:
            stk.append([start_index, index])
            stk.append([index, last_index])

    return dists

def sketch2array(sketch):
    sketch_array = [np.array(s + [[0]*(len(s[0])-1) + [1]]) for s in sketch]
    point = np.concatenate([s.transpose() for s in sketch_array])
    return point

def array2sketch(sketch_array):
    end_idxs = np.argwhere(sketch_array[:,3]==1).reshape(-1) + 1
    start_idxs = end_idxs[:-1]
    start_idxs = np.insert(start_idxs, 0, 0)
    sketch = [sketch_array[s:e,:3].transpose().tolist() for s, e in zip(start_idxs, end_idxs)]
    # assert len(sketch) > 0, 'error:{},{}'.format(start_idxs, end_idxs)
    return sketch

def double_stroke(stroke):
    new_stroke = [[stroke[0][0]], [stroke[1][0]], [stroke[2][0]]]
    for i in range(1, len(stroke[0])):
        # x = (stroke[0][i-1] + stroke[0][i] + np.random.randint(-1, 1)) // 2
        # y = (stroke[1][i-1] + stroke[1][i] + np.random.randint(-1, 1)) // 2
        x = (stroke[0][i-1] + stroke[0][i] + np.random.randint(0, 1)) // 2
        y = (stroke[1][i-1] + stroke[1][i] + np.random.randint(0, 1)) // 2
        new_stroke[0].extend([x, stroke[0][i]])
        new_stroke[1].extend([y, stroke[1][i]])
        new_stroke[2].extend([stroke[2][i]]*2)
    return new_stroke

def double_sketch(sketch):
    for i in range(len(sketch)):
        sketch[i] = double_stroke(sketch[i])

def check_single_point(sketch):
    stroke_lens = np.array([len(s[0]) for s in sketch])
    assert (stroke_lens > 1).all(), 'Stroke Len:{}'.format(stroke_lens)


def sample(sketch, N):
    # point_num = cal_point_number(sketch)

    while cal_point_number(sketch) < N:
        double_sketch(sketch)

    # point_num_double = cal_point_number(sketch)

    sketch_array = sketch2array(sketch)

    end_idxs = np.argwhere(sketch_array[:,3]==1).reshape(-1)
    start_idxs = end_idxs[:-1] + 1
    start_idxs = np.insert(start_idxs, 0, 0)
    idxs = list(zip(start_idxs, end_idxs))

    dists = rdp_iter(sketch_array[:,:2], idxs, epsilon=0.0)
    idxs = np.argsort(-dists)[:N]
    idxs = sorted(idxs)
    new_sketch_array = sketch_array[idxs]

    new_sketch = array2sketch(new_sketch_array)

    # new_point_num = cal_point_number(new_sketch)
    # assert new_point_num == 256, 'Error:{},{},{}'.format(point_num, point_num_double, new_point_num)
    check_single_point(new_sketch)

    return new_sketch

if __name__ == "__main__":
    zoom_times = 2
    sample_n = 256
    category_name = 'airplane'
    raw_path = os.path.join('..', 'data', 'grouping_sample', '{}_raw.ndjson'.format(category_name))
    with open(raw_path, 'rb') as f:
        raw_data = ndjson.load(f)

    for data in raw_data:
        sketch = data['drawing']
        canvas = drawColor(sketch, [int(256*zoom_times)]*2, zoom_times)
        cv.imshow('Test Sample', canvas)
        key_num = cv.waitKey(0)
        sampled_sketch = sample(sketch, sample_n)
        canvas = drawColor(sampled_sketch, [int(256*zoom_times)]*2, zoom_times)
        cv.imshow('Test Sample', canvas)
        key_num = cv.waitKey(0)
        if key_num == ord('q'):
            break