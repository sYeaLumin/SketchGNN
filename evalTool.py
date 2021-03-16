import os
import json
import math
import numpy as np

def eval_unalign_batch1(model, loader):
    """
    batch_size must be 1
    return:
    predictList: (N, Px)
    lossList:    (N, )
    """
    predictList = [] # (N, Px)
    lossList = []
    # model.clear_confusion()
    for data in loader:
        loss, out = model.test(data, if_eval=True)
        lossList.append(loss.item())
        predictList.append(out.tolist())

    return predictList, lossList

def eval_align_batchN(model, loader, P=256):
    """
    return:
    predictList: (N, P) e.g. (N, 256)
    lossList:    (B, )
    """
    predictList = [] # sample_num x point_num(256)
    lossList = []
    # model.clear_confusion()
    for data in loader:
        loss, out = model.test(data, if_eval=True)
        lossList.append(loss.item())
        predictList.extend(out.reshape(-1, P).tolist())

    return predictList, lossList


def get_eval_result(test_data, predict):
    for i, data in enumerate(test_data):
        predict_result = predict[i]
        sketch = data['drawing']
        for stroke in sketch:
            label_num = len(stroke[2])
            stroke[2] = predict_result[:label_num]
            predict_result = predict_result[label_num:]

    return test_data


def eval_without_len(testData, predict):
    p_metric_list = []
    c_metric_list = []
    for i, data in enumerate(testData):
        predict_result = predict[i] # 256
        p_right = 0
        p_sum = len(predict_result)

        sketch = data['drawing']
        c_right = 0
        c_sum = len(sketch)
        for j, stroke in enumerate(sketch):
            stroke_label_true = stroke[2]
            stroke_label_predict = np.array(predict_result[:len(stroke_label_true)])
            predict_result = np.array(predict_result[len(stroke_label_true):])
            p_right += np.sum(stroke_label_predict==stroke_label_true)
            if np.average(stroke_label_predict==stroke_label_true) > 0.75:
                c_right += 1

        p_metric_list.append(p_right / p_sum)
        c_metric_list.append(c_right / c_sum) 

    return p_metric_list, c_metric_list

def eval_with_len(testData, predict):
    p_metric_list = []
    c_metric_list = []
    for i, data in enumerate(testData):
        predict_result = predict[i] # 256
        p_right = 0
        p_sum = 0

        sketch = data['drawing']
        c_right = 0
        c_sum = 0
        # c_sum = len(sketch)
        for stroke in sketch:
            if stroke[2][0] == -1:
                continue

            c_sum += 1
            
            stroke_len = [1]
            for j in range(1, len(stroke[0])):
                stroke_len.append(int(math.sqrt(pow(stroke[0][j]-stroke[0][j-1],2) + 
                                            pow(stroke[1][j]-stroke[1][j-1],2))))
            stroke_len = np.array(stroke_len)
            stroke_p_sum = np.sum(stroke_len)
            p_sum += stroke_p_sum
            stroke_label_true = stroke[2]
            stroke_label_predict = np.array(predict_result[:len(stroke_label_true)])
            predict_result = np.array(predict_result[len(stroke_label_true):])
            right_index = np.array(stroke_label_predict==stroke_label_true, dtype=np.int)
            stroke_p_right = np.sum(right_index*stroke_len)
            p_right += stroke_p_right
            if stroke_p_right / stroke_p_sum > 0.75:
                c_right += 1

        p_metric_list.append(p_right / p_sum)
        c_metric_list.append(c_right / c_sum) 
        
    return p_metric_list, c_metric_list


if __name__ == "__main__":
    forms = 'score:{0[0]:.5}\tloss_avg:{0[1]:.3}\tP_metric:{0[2]:.3}%\tC_metric:{0[3]:.3}%\tbest_e:{0[4]}\t{0[5]}'
    lists = [
        [0.641, 0.157412, 0.65,0.522, 1, {'a':1,'n':2}],
        [0.444, 0.1572, 0.7865,0.25, 1, {'a':1,'n':4}],
        [0.141, 0.17412, 0.6735,0.53, 1, {'a':6,'n':2}]
    ]
    # write_txt('log', 'aaa', forms, lists)
