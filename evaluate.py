import os
import ndjson
import json
import time
from options import TestOptions
from framework import SketchModel
from utils import load_data
from writer import Writer
import numpy as np
from evalTool import *

def run_eval(opt=None, model=None, loader=None, dataset='test', write_result=False):
    if opt is None:
        opt = TestOptions().parse()
    if model is None:
        model = SketchModel(opt)
    if loader is None:
        loader = load_data(opt, datasetType=dataset, permutation=opt.permutation)
    # print(len(loader)) 
    if opt.eval_way == 'align':
        predictList, lossList = eval_align_batchN(model, loader, P=opt.points_num)
    elif opt.eval_way == 'unalign':
        predictList, lossList = eval_unalign_batch1(model, loader)
    else:
        raise NotImplementedError('eval_way {} not implemented!'.format(opt.eval_way))
    # print(predictList.shape)
    testData = []
    with open(os.path.join('data', opt.dataset, 'train', 
            '{}_{}.ndjson'.format(opt.class_name, dataset)), 'r') as f:
        testData = ndjson.load(f)

    if opt.metric_way == 'wlen':
        p_metric_list, c_metric_list = eval_with_len(testData, predictList)
    elif opt.metric_way == 'wolen':
        p_metric_list, c_metric_list = eval_without_len(testData, predictList)
    else:
        raise NotImplementedError('metric_way {} not implemented!'.format(opt.metric_way))

    if write_result:
        testData = get_eval_result(testData, predictList)
        result_path = os.path.join('data', opt.dataset, 'train', '{}_{}.ndjson'.format(opt.class_name, 'res'))
        with open(result_path, 'w') as f:
            ndjson.dump(testData, f)
    
    loss_avg = np.average(lossList)
    P_metric = np.average(p_metric_list)
    C_metric = np.average(c_metric_list)
    # print('P_metric:{:.4}%\tC_metric:{:.4}%'.format(P_metric*100, C_metric*100))

    return loss_avg, P_metric, C_metric




if __name__ == "__main__":
    _, P_metric, C_metric = run_eval(write_result=True)
    print('P_metric:{:.4}%\tC_metric:{:.4}%'.format(P_metric*100, C_metric*100))


