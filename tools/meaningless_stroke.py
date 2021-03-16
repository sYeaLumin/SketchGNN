import os
import sys
sys.path.append(os.path.dirname(__file__))

import ndjson
import numpy as np
import cv2 as cv
from center_sketch  import centeredSketch
from sample_sketch import sketch2array, array2sketch, sample
import argparse

def generate_meaningless_stroke(point_num, speed=5):
    points = np.random.randint(0, 255, (1, 2))
    stroke_direct = np.random.randint(-speed, speed, (1, 2))
    direct_speed = max(speed, 4)
    while point_num > 0:
        last_point = points[-1]
        new_point = last_point + stroke_direct
        p = new_point[0] / 256
        p *= abs(p)

        x_low, x_high = int(-direct_speed * p[0]), int(direct_speed * (1-p[0]))
        y_low, y_high = int(-direct_speed * p[1]), int(direct_speed * (1-p[1]))
        stroke_direct_x = np.random.randint(x_low, x_high)
        stroke_direct_y = np.random.randint(y_low, y_high)
        stroke_direct += np.array([stroke_direct_x, stroke_direct_y])

        points = np.append(points, new_point, axis=0)
        point_num -= 1
    return points

def cal_stroke_len(stroke):
    points = np.array(stroke[:2])
    points_s = points[:, :-1]
    points_e = points[:, 1:]
    offset = points_e - points_s
    offset = offset * offset
    stroke_len = np.sum(np.sqrt(offset[0] + offset[1]))
    return stroke_len

def additional_label(sketch, percent, label_num=-1, labeling_type='random'):
    return centered(sample(meaningless(sketch, percent, label_num, labeling_type), N=256))

def meaningless(sketch, percent, label_num=-1, labeling_type='random'):
    if percent == 0:
        return sketch

    point_num = sum(list(map(lambda x:len(x[0]), sketch)))
    meaningless_point_num = int(point_num * percent)
    # print(meaningless_point_num)

    stroke_lens = [cal_stroke_len(s) for s in sketch]
    draw_speed = int(sum(stroke_lens) / point_num) + 1
    # print('draw_speed', draw_speed)

    points = generate_meaningless_stroke(meaningless_point_num, speed=draw_speed)
    new_stroke = points.transpose().tolist()

    # assign label
    if label_num > 0:
        if labeling_type == 'random':
            meaningless_label = np.random.randint(label_num)
        elif labeling_type == 'addition':
            meaningless_label = label_num
        else:
            raise NotImplementedError('Labeling Type {} is not implemented!'.format(labeling_type))
    else:
        meaningless_label = -1
    new_stroke.append([meaningless_label]*len(new_stroke[0]))
    sketch.append(new_stroke)

    return sketch

def centered(sketch):
    label = list(map(lambda s:s[2][0], sketch))
    sketch = centeredSketch(sketch)
    for i, stroke in enumerate(sketch):
        stroke.append([label[i]]*len(stroke[0]))
    return sketch

if __name__ == "__main__":
    from draw_sketch import drawColor
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='grouping_percent', help='')
    parser.add_argument('--class-name', type=str, default='airplane', help='name of the class')
    parser.add_argument('--percent', type=float, default=0.5)
    parser.add_argument('--calculate', action='store_true', help='')
    parser.add_argument('--cal-dataset', type=str, default='test', help='')
    args = parser.parse_args()

    zoom_times = 2
    sample_n = 256
    category_name = args.class_name
    ori_path = os.path.join('..', 'data', 'grouping_meanless', '{}_raw.ndjson'.format(category_name))
    with open(ori_path, 'rb') as f:
        ori_data = ndjson.load(f)

    seeds = int(args.percent*100 + 400)
    np.random.seed(seeds)

    if args.calculate:
        if args.cal_dataset == 'test':
            test_data = ori_data[-100:]
            test_data_path = os.path.join('..', 'data', 'grouping_meanless', 'train', '{}_test.ndjson'.format(category_name))
            for data in test_data:
                data['drawing'] = centered(sample(meaningless(centered(data['drawing']), args.percent), N=256))
        elif args.cal_dataset == 'train':
            test_data = ori_data[:-150]
            test_data_path = os.path.join('..', 'data', 'grouping_meanless', 'train', '{}_train.ndjson'.format(category_name))
            for data in test_data:
                data['drawing'] = centered(data['drawing'])
        elif args.cal_dataset == 'valid':
            test_data = ori_data[-150:-100]
            test_data_path = os.path.join('..', 'data', 'grouping_meanless', 'train', '{}_valid.ndjson'.format(category_name))
            for data in test_data:
                assert len(data['drawing'][0]) == 3, 'Error:{}, {}'.format(data['i'], data['key_id'])
                data['drawing'] = centered(sample(data['drawing'], N=256))
        else:
            raise NotImplementedError('.')
            
        with open(test_data_path, 'w') as f:
            ndjson.dump(test_data, f)
    else:
        # for data in ori_data[-100:]:
        for data in ori_data:
            sketch = centered(data['drawing'])
            canvas0 = drawColor(sketch, [int(256*zoom_times)]*2, zoom_times)
            sketch = meaningless(sketch, args.percent, label_num=-1)
            canvas1 = drawColor(sketch, [int(256*zoom_times)]*2, zoom_times)
            sketch = centered(sample(sketch, N=256))
            canvas2 = drawColor(sketch, [int(256*zoom_times)]*2, zoom_times)
            canvas = np.concatenate((canvas0, canvas1, canvas2), axis=1)

            cv.imshow('Meaningless Stroke', canvas)
            key_num = cv.waitKey(0)
            if key_num == ord('q'):
                break