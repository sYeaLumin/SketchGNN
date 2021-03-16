import os
import time
import torch
import json
import argparse
import numpy as np
import random

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
    
    def initialize(self):
        # data params
        self.parser.add_argument('--class-name', type=str, required=True, 
                                 help='the name of the class to train or test')
        self.parser.add_argument('--points-num', type=int, default=256, 
                                 help='the number of the points in one sketch')
        self.parser.add_argument('--dataset', type=str, required=True, 
                                 help='the name of dataset')
        self.parser.add_argument('--perm-arg', type=float, default=0)
        self.parser.add_argument('--perm-type', type=str, default='-')
        # networks
        self.parser.add_argument('--net-name', type=str, default='SketchTwoLine', 
                                 help='the name of the net to use')
        self.parser.add_argument('--in-feature', type=int, default=2, 
                                 help='the number of the feature input to net')
        self.parser.add_argument('--out-segment', type=int, required=True, 
                                 help='the number of the labels of the segment')
        self.parser.add_argument('--batch-size', type=int, default=16, 
                                 help='intout batch size')
        self.parser.add_argument('--alpha', type=float, default=0.2, 
                                 help='the alpha of leakyrelu')
        # for net work
        self.parser.add_argument('--mlp-segment', type=int, nargs='+', default=[128, 64],
                                 help='the number of hidden of layers')
        self.parser.add_argument('--adj-type', type=str, default='dynamic',
                                 help='static or dynamic')
        self.parser.add_argument('--local-adj-type', type=str, default='static',
                                 help='static or dynamic')
        self.parser.add_argument('--global-adj-type', type=str, default='dynamic',
                                 help='static or dynamic')
        self.parser.add_argument('--kernel-size', type=int, default=8,
                                 help='')     
        self.parser.add_argument('--dilation', type=int, default=16,
                                 help='')
        self.parser.add_argument('--stochastic', action='store_true', 
                                 help='')
        self.parser.add_argument('--epsilon', type=float, default=0.1,
                                 help='')                                    
        ## --adj-type dynamic
        self.parser.add_argument('--gcn-type', type=str, default='edge',
                                 help='edge or mr')        
        self.parser.add_argument('--global-k', type=int, default=8,
                                 help='')
        self.parser.add_argument('--global-dilation', type=int, default=16,
                                 help='')
        self.parser.add_argument('--global-stochastic', type=bool, default=True,
                                 help='')
        self.parser.add_argument('--global-epsilon', type=float, default=0.1,
                                 help='')
        self.parser.add_argument('--local-k', type=int, default=8,
                                 help='')
        self.parser.add_argument('--local-dilation', type=int, default=16,
                                 help='')
     
        self.parser.add_argument('--local-stochastic', type=bool, default=True,
                                 help='')
        self.parser.add_argument('--local-epsilon', type=float, default=0.1,
                                 help='')
        self.parser.add_argument('--mixedge', type=bool, default=True,
                                 help='')
        ## --gcn-type edge or mr
        self.parser.add_argument('--aggr-type', type=str, default='max',
                                 help='max or mean or sum')
        self.parser.add_argument('--act-type', type=str, default='relu',
                                 help='relu or leakyrelu or prelu')
        self.parser.add_argument('--norm-type', type=str, default='None',
                                 help='None or batch or instance')
        self.parser.add_argument('--block-type', type=str, default='res',
                                 help='basic or res or dense')
        self.parser.add_argument('--fusion-type', type=str, default='mix',
                                 help='max or sketch or mix')
        self.parser.add_argument('--n-blocks', type=int, default=3,
                                 help='')
        self.parser.add_argument('--channels', type=int, default=32,
                                 help='channel in backbone')
        self.parser.add_argument('--pool-channels', type=int, default=256,
                                 help='')            
        # general params
        self.parser.add_argument('--seed', type=int,
                                 help='if specified, uses seed')
        self.parser.add_argument('--gpu-ids', type=str, default='0', 
                                 help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints-dir', type=str, default='checkpoints', 
                                 help='models are saved here')
        self.parser.add_argument('--pretrain', type=str, default='-', 
                                 help='which pretrain model to load')
        self.parser.add_argument('--which-epoch', type=str, default='latest', 
                                 help='which epoch to load? set to latest to use latest cached model')
        # eval params
        self.parser.add_argument('--eval-way', type=str, default='align', 
                                 help='align or unalign')
        self.parser.add_argument('--metric-way', type=str, default='wlen', 
                                 help='wlen or wolen')

        # other
        self.parser.add_argument('--num-workers', type=int, default=0,
                                 help='')

    def parse(self, params=None):
        if not self.initialized:
            self.initialize()
        self.opt, _ = self.parser.parse_known_args()
        self.opt.is_train = self.is_train

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])
        
        if self.opt.seed is not None:
            # import numpy as np
            # import random
            # torch.manual_seed(self.opt.seed)
            # np.random.seed(self.opt.seed)
            # random.seed(self.opt.seed)
            set_seed(self.opt.seed)
        
        # change opt from params
        if params:
            for key in params.keys():
                setattr(self.opt, key, params[key])
        
        args = vars(self.opt)
        if self.is_train:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

            self.opt.timestamp = time.strftime("%b%d_%H_%M")
            # save to the disk
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.dataset, self.opt.class_name, self.opt.timestamp)
            mkdir(expr_dir)

            # option record
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')

        return self.opt

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--epoch', type=int, default=100,
                                 help='epoch')
        self.parser.add_argument('--lr', type=float, default=0.002, 
                                 help='initial learning rate for adam')
        self.parser.add_argument('--beta1', type=float, default=0.9, 
                                 help='momentum term of adam')
        self.parser.add_argument('--lr-policy', type=str, default='step', 
                                 help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr-decay-iters', type=int, default=80, 
                                 help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument("--shuffle", action='store_true',
                                 help='if shuffle dataset while training')
        self.parser.add_argument("--disorder-id", action='store_true', default=False,
                                 help='if shuffle feature3-id')
        self.parser.add_argument("--permutation", action='store_true',
                                 help='if data permutation')
        self.is_train = True
        self.parser.add_argument("--trainlist", type=str, default='-',
                                 help='the json file name of list to train')
        self.parser.add_argument("--random-iter", type=int, default=1,
                                 help='') 
        self.parser.add_argument("--train-log-name", type=str, default='-',
                                 help='')                  

        self.parser.add_argument("--train-dataset", type=str, default='train',
                                 help='which dataset to train, train or train2 or trainxxx')
        self.parser.add_argument("--valid-dataset", type=str, default='test',
                                 help='which dataset to valid during training')
        self.parser.add_argument("--comment", type=str, default='-',
                                 help='some comments')

        self.parser.add_argument('--plot-freq', type=int, default=1, 
                                 help='frequency of ploting training loss')
        self.parser.add_argument('--print-freq', type=int, default=10, 
                                 help='frequency of showing training loss on console')
        self.parser.add_argument('--save-epoch-freq', type=int, default=20, 
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--run-test-freq', type=int, default=1, 
                                 help='frequency of running test in training script')

        self.parser.add_argument('--no-vis', action='store_true', 
                                 help='will not use tensorboard')
        self.parser.add_argument('--plot-weights', action='store_true', 
                                 help='plots network weights, etc.')


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--timestamp', type=str, default='-', 
                help='the timestep of the model')
        self.parser.add_argument('--print-freq', type=int, default=2, 
                help='frequency of showing training results on console')
        self.parser.add_argument("--permutation", action='store_true',
                                 help='if data permutation')
        self.is_train = False



if __name__ == "__main__":
    import numpy as np
    from sklearn.model_selection import ParameterSampler

    _opt = TrainOptions().parse()
    with open(_opt.randomlist, 'r') as f:
        param_grid = json.load(f)
    rng = np.random.RandomState(0)
    param_list = list(ParameterSampler(param_grid, n_iter=5, random_state=rng))

    for params in param_list:
        opt = TrainOptions().parse(params)
        print(params)
