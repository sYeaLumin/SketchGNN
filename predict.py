import os
import json
import time
import numpy as np
from options import TestOptions
from framework import SketchModel
from utils import load_data
from writer import Writer

if __name__ == "__main__":
    opt = TestOptions().parse()
    model = SketchModel(opt)
    loader = load_data(opt, datasetType='test')
    writer = Writer(opt)
    
    predict = np.array([])
    for i, data in enumerate(loader):
        loss, out = model.test(data, if_eval=True)
        predict = np.append(predict, out)
    
    i = 0
    predictData = []
    with open(os.path.join('data', opt.dataset, 'train', 
            '{}_{}.ndjson'.format(opt.class_name, 'test')), 'r') as f:
        for line in f:
            data = json.loads(line)
            sketch = data["drawing"]
            for stroke in sketch:
                l = len(stroke[0])
                stroke[2] = predict[i:i+l].astype(np.int32).tolist()
                i += l
            data["drawing"] = sketch
            predictData.append(data)
    
    with open(os.path.join('data', opt.dataset, 'train', 
            '{}_{}.ndjson'.format(opt.class_name, 'res')), 'w') as f:
        for data in predictData:
            json.dump(data, f)
            f.write("\n")
