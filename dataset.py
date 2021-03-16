import os.path as osp
import copy
import json
import math
import random
import torch
import numpy as np
from torch_geometric.data import Data
from blocks.conv import cal_edge_attr
from tools.cut_stroke import cutoff
from tools.dislocation import dislocate
from tools.meaningless_stroke import additional_label
from tools.center_sketch import rotate_theta, add_normal_noise

class SketchData(Data):
    def __init__(self, stroke_idx=None, x=None, edge_index=None, edge_attr=None, y=None,
                 pos=None, norm=None, face=None, **kwargs):
        super(SketchData, self).__init__(x, edge_index, edge_attr, y, pos, norm, face, **kwargs)
        self.stroke_idx = stroke_idx
        self.stroke_num = max(stroke_idx) + 1

    def __inc__(self, key, value):
        if 'index' in key or 'face' in key:
            return self.num_nodes
        elif 'stroke' in key:
            return self.stroke_num
        else:
            return 0

class SketchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, root, class_name, split='train', permutation=False):
        self.class_name = class_name
        self.split = split
        self.pt_dir = osp.join(root, '{}_{}.pt'.format(self.class_name, self.split))
        self.json_dir = osp.join(root, '{}_{}.ndjson'.format(self.class_name, self.split))
        self.permutation = permutation
        self.perm_args = opt.perm_arg
        self.perm_type = opt.perm_type
        self.out_segment = opt.out_segment

        if self.permutation:
            if opt.is_train:
                if self.perm_type == 'dislocation':
                    self.processed_data = self._process()
                    self.augmentation(perm_args=[0.1, 0.3]) # dislocation
                elif self.perm_type == 'cutoff':
                    self.processed_data = self._process()
                    self.augmentation(perm_args=[0.8, 3.2]) # cut off 
                elif self.perm_type[:8] == 'meanless':
                    self.processed_data = self._process(perm_arg=0)
                    self.augmentation(perm_args=[0.15, 0.2]) # meanless
                else:
                    raise NotImplementedError('Permutation Type {} is not implemented!'.format(self.perm_type))
            else:
                self.processed_data = self._process(opt.perm_arg)
        else:
            if osp.exists(self.pt_dir):
                self.processed_data = torch.load(self.pt_dir)
            else:
                self.processed_data = self._process()

    def __getitem__(self, index):
        return self.processed_data[index]

        # if self.permutation:
        #     # dislocation
        #     # sketch = self.processed_data[index]
        #     sketch = copy.deepcopy(self.processed_data[index])
        #     dislocate_stroke_num = int(sketch.stroke_num*0.5)
        #     idxs = np.random.choice(sketch.stroke_num, dislocate_stroke_num, replace=False)
        #     for idx in idxs:
        #         _idx = (sketch.stroke_idx == idx).int().reshape(-1, 1)
        #         offset = self.perm_args*(torch.rand(2) - 0.5)
        #         offset = offset.repeat(256, 1) * _idx
        #         sketch.x += offset

        #     # # noise
        #     # sketch = copy.deepcopy(self.processed_data[index])
        #     # # _sketch = self.processed_data[index]
        #     # sketch.x = sketch.x*255 + torch.randn(sketch.x.shape)*self.perm_args
        #     # # sketch_x = _sketch.x + torch.randn(_sketch.x.shape)*self.perm_args
            
        #     # # rotation
        #     # sketch = self.processed_data[index]
        #     # theta = random.randint(0, self.perm_args) * math.pi/180
        #     # if random.randint(0, 1):
        #     #     theta = -theta
        #     # rotate_matrix = torch.tensor([[math.cos(theta), math.sin(theta)],[-math.sin(theta), math.cos(theta)]])
        #     # sketch.x = torch.matmul(sketch.x, rotate_matrix.to(sketch.x.dtype).to(sketch.x.device))

        #     # norm
        #     max_point = torch.max(sketch.x, dim=0).values
        #     min_point = torch.min(sketch.x, dim=0).values
        #     sketch.x = (sketch.x - min_point) / (max_point - min_point)
        #     return sketch
        # else:
        #     return self.processed_data[index]
    
    def __len__(self):
        return len(self.processed_data)
    
    def augmentation(self, perm_args):
        for perm_arg in perm_args:
            self.processed_data.extend(self._process(perm_arg))
    
    def _process(self, perm_arg=None):
        if perm_arg is not None:
            print('Processing with augment param {} ...'.format(perm_arg))
        else:
            print('Processing without augment param ...')

        raw_data = []
        with open(self.json_dir, 'r') as f:
            for line in f:
                raw_data.append(json.loads(line)["drawing"])
        processed_data = []

        for sketch in raw_data:
            if perm_arg is not None:
                if self.perm_type == 'rotate':
                    theta = perm_arg * math.pi/180
                    sketch = rotate_theta(sketch, theta)
                elif self.perm_type == 'noise':
                    sketch = add_normal_noise(sketch, scale=perm_arg)
                elif self.perm_type == 'dislocation':
                    sketch = dislocate(sketch, 1, perm_arg)
                elif self.perm_type == 'cutoff':
                    sketch = cutoff(sketch, perm_arg)
                elif self.perm_type == 'meanless_addition':
                    sketch = additional_label(sketch, perm_arg, label_num=self.out_segment-1, labeling_type='addition')
                elif self.perm_type == 'meanless_random':
                    sketch = additional_label(sketch, perm_arg, label_num=self.out_segment, labeling_type='random')
                else:
                    raise NotImplementedError('Permutation Type {} is not implemented!'.format(self.perm_type))

            sketchArray = [np.array(s) for s in sketch]
            stroke_idx = np.concatenate([np.zeros(len(s[0])) + i for i, s in enumerate(sketchArray)])
            point = np.concatenate([s.transpose()[:,:2] for s in sketchArray])
            # normalize the data (N x 2)
            point = point.astype(np.float)
            max_point = np.max(point, axis=0)
            min_point = np.min(point, axis=0)
            point = (point - min_point) / (max_point - min_point)
            # point /= 255
            
            # label: c (N,)
            label = np.concatenate([s[2] for s in sketchArray], axis=0) # (N, )

            # edge_index
            edge_index = []
            s = 0
            for stroke in sketchArray:
                # edge_index.append([s,s])
                for i in range(len(stroke[0])-1):
                    edge_index.append([s+i, s+i+1])
                    edge_index.append([s+i+1, s+i])
                # edge_index.append([s,s+len(stroke[0])-1])
                s += len(stroke[0])
            edge_index = np.array(edge_index).transpose()

            sketch_data = SketchData(x=torch.FloatTensor(point), 
                                    edge_index=torch.LongTensor(edge_index),
                                    stroke_idx=torch.LongTensor(stroke_idx),
                                    y=torch.LongTensor(label))
            sketch_data.edge_attr = cal_edge_attr(sketch_data.edge_index, sketch_data.x)
            processed_data.append(sketch_data)
        
        # torch.save(processed_data, self.pt_dir)
        return processed_data


if __name__ == "__main__":
    import os
    import torch
    from options import TestOptions
    from utils import load_data
    opt = TestOptions().parse()
    opt.perm_arg = 0.3
    dataloader = load_data(opt, datasetType='test', permutation=True)
    for e in range(3):
        for i, batch in enumerate(dataloader):
            print(batch.x)
            # print(batch.stroke_idx)
            # print(batch.stroke_idx[:300])
            # print(batch)
            # print(batch.num_graphs)
            # print(batch.batch.shape)
            # row, col = batch.edge_index
            # offset = batch.pos[col] - batch.pos[row]
            # print(offset)
            # dist = torch.norm(offset, p=2, dim=-1).view(-1, 1)
            # edge_attr = torch.cat([offset, dist], dim=-1)
            # print(batch.edge_attr)
            break

