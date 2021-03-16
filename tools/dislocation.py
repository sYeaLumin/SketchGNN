import os
import sys
sys.path.append(os.path.dirname(__file__))

import ndjson
import numpy as np
import cv2 as cv
from center_sketch import centeredSketch
import argparse

def set_seed(seed):
    np.random.seed(seed)

def dislocate_stroke(stroke, ranges):
    if ranges == 0:
        return
    random_range = int(256*ranges)
    offset_x = np.random.randint(-random_range, random_range)
    offset_y = np.random.randint(-random_range, random_range)
    stroke[0] = list(map(lambda x:x + offset_x, stroke[0]))
    stroke[1] = list(map(lambda y:y + offset_y, stroke[1]))
    # print(offset_x, offset_y)

def dislocate(sketch, percent, ranges):
    # cal stroke num
    stroke_num = len(sketch)
    dislocate_stroke_num = int(stroke_num * percent)
    # print('Dis:', dislocate_stroke_num, stroke_num)

    # select strokes to dislocate
    idxs = np.random.choice(stroke_num, dislocate_stroke_num, replace=False)

    # dislocate
    for idx in idxs:
        # print(idx, ':')
        dislocate_stroke(sketch[idx], ranges)

    # return sketch

    # new data
    label = list(map(lambda s:s[2][0], sketch))
    sketch = centeredSketch(sketch)
    for i, stroke in enumerate(sketch):
        stroke.append([label[i]]*len(stroke[0]))

    return sketch

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    from draw_sketch import drawColor, writesvg, color_list

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='grouping_dislocation', help='')
    parser.add_argument('--class-name', type=str, default='airplane', help='name of the class')
    parser.add_argument('--percent', type=float, default=0.5)             
    parser.add_argument('--range', type=float, default=0.25)
    parser.add_argument('--calculate', action='store_true', help='')
    parser.add_argument('--seed', type=int, default=424)  

    args = parser.parse_args()

    zoom_times = 2
    sample_n = 256
    category_name = args.class_name
    ori_path = os.path.join('..', 'data', 'grouping_dislocation', '{}256.ndjson'.format(category_name))
    with open(ori_path, 'rb') as f:
        ori_data = ndjson.load(f)

    # seeds = int(args.range*100 + 400)
    # set_seed(seeds)
    set_seed(args.seed)

    # label
    label_name_list = []
    with open(os.path.join('..', 'data', args.dataset, '{}.txt'.format(args.class_name)), 'r') as f:
        for line in f:
            label_name_list.append(line[:-1])

    if args.calculate:
        test_data = ori_data[-100:]
        for data in test_data:
            data['drawing'] = dislocate(data['drawing'], args.percent, args.range)

        test_data_path = os.path.join('..', 'data', 'grouping_dislocation', 'train', '{}_test.ndjson'.format(category_name))
        with open(test_data_path, 'w') as f:
            ndjson.dump(test_data, f)
    else:
        for data in ori_data[-100:]:
        # for data in ori_data:
            sketch = data['drawing']
            canvas0 = drawColor(sketch, [int(256*zoom_times)]*2, zoom_times)
            sketch = dislocate(sketch, args.percent, args.range)
            canvas1 = drawColor(sketch, [int(256*zoom_times)]*2, zoom_times)
            canvas = np.concatenate((canvas0, canvas1), axis=1)

            cv.imshow('Dislocation', canvas)
            key_num = cv.waitKey(0)
            if key_num == ord('q'):
                break
            elif key_num == ord('w'):
                mkdir(os.path.join('visuals', args.dataset))
                writesvg(sketch, label_name_list, os.path.join('visuals', args.dataset,'{}_{}_{}.svg'.format(args.class_name, 'dislocation', args.range)), color_list)
                # key_num = cv.waitKey(0)