import os
import sys
sys.path.append(os.path.dirname(__file__))


import ndjson
import numpy as np
import cv2 as cv
import argparse

def set_seed(seed):
    np.random.seed(seed)

def cutoff_stroke(stroke, per_stroke_len):
    # assert per_stroke_len > 1
    per_stroke_len = max(per_stroke_len, 2)

    ori_stroke = np.array(stroke)

    new_strokes = []
    while ori_stroke.any():
        if ori_stroke.shape[1] - per_stroke_len == 1:
            per_stroke_len_cur = per_stroke_len + 1
        else:
            per_stroke_len_cur = per_stroke_len

        new_stroke = ori_stroke[:,:per_stroke_len_cur]

        new_strokes.append(new_stroke.tolist())
        ori_stroke = ori_stroke[:,per_stroke_len_cur:]

    return new_strokes


def cutoff(sketch, cut_range):
    point_num = sum(list(map(lambda x:len(x[0]), sketch)))
    # per_stroke_len = int(point_num * cut_range / len(sketch))
    per_stroke_len = int(point_num / len(sketch) / cut_range)

    new_sketch = []
    for stroke in sketch:
        new_sketch.extend(cutoff_stroke(stroke, per_stroke_len))

    return new_sketch

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    from draw_sketch import drawColor, writesvg, color_list

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='grouping_cutoff', help='')
    parser.add_argument('--class-name', type=str, default='airplane', help='name of the class')  
    parser.add_argument('--range', type=float, default=0.25)
    parser.add_argument('--calculate', action='store_true', help='')
    parser.add_argument('--seed', type=int, default=424)  

    args = parser.parse_args()

    zoom_times = 2
    sample_n = 256
    category_name = args.class_name
    ori_path = os.path.join('..', 'data', 'grouping_cutoff', '{}256.ndjson'.format(category_name))
    with open(ori_path, 'rb') as f:
        ori_data = ndjson.load(f)
    
    # label
    label_name_list = []
    with open(os.path.join('..', 'data', args.dataset, '{}.txt'.format(args.class_name)), 'r') as f:
        for line in f:
            label_name_list.append(line[:-1])


    if args.calculate:
        test_data = ori_data[-100:]
        for data in test_data:
            data['drawing'] = cutoff(data['drawing'], args.range)

        test_data_path = os.path.join('..', 'data', 'grouping_cutoff', 'train', '{}_test.ndjson'.format(category_name))
        with open(test_data_path, 'w') as f:
            ndjson.dump(test_data, f)
    else:
        for data in ori_data[-100:]:
        # for data in ori_data:
            sketch = data['drawing']
            canvas0 = drawColor(sketch, [int(256*zoom_times)]*2, zoom_times)
            sketch = cutoff(sketch, args.range)
            canvas1 = drawColor(sketch, [int(256*zoom_times)]*2, zoom_times)
            canvas = np.concatenate((canvas0, canvas1), axis=1)

            cv.imshow('cutoff', canvas)
            key_num = cv.waitKey(0)
            if key_num == ord('q'):
                break
            elif key_num == ord('w'):
                mkdir(os.path.join('visuals', args.dataset))
                writesvg(sketch, label_name_list, os.path.join('visuals', args.dataset,'{}_{}_{}.svg'.format(args.class_name, 'cut', args.range)), color_list, ifendpoint=True)