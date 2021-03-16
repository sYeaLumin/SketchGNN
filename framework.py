import os
import torch
from torch.optim import lr_scheduler
import numpy as np
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, adjusted_rand_score
import net
import utils

class SketchModel:
    def __init__(self, opt):
        self.opt = opt
        self.is_train = opt.is_train
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.dataset, opt.class_name, opt.timestamp)
        self.pretrain_dir = os.path.join(opt.checkpoints_dir, opt.dataset, opt.class_name, opt.pretrain)
        self.optimizer = None
        self.loss_func = None
        self.loss = None
        self.confusion = None # confusion matrix
        self.multi_confusion = None

        self.net_name = opt.net_name
        self.net = net.init_net(opt)
        self.net.train(self.is_train)
        self.loss_func = torch.nn.NLLLoss().to(self.device)

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(), 
                                              lr=opt.lr, 
                                              betas=(opt.beta1, 0.999))
            self.scheduler = utils.get_scheduler(self.optimizer, opt)
        
        if not self.is_train: #or opt.continue_train:
            self.load_network(opt.which_epoch, mode='test')
            
        if self.is_train and opt.pretrain != '-':
            self.load_network(opt.which_epoch, mode='pretrain')
    
    def forward(self, x, edge_index, data):
        out = self.net(x, edge_index, data)
        return out

    def backward(self, out, label):
        """
        out: (B*N, C)
        label: (B*N, )
        """
        self.loss = self.loss_func(out, label)
        self.loss.backward()

    def step(self, data):
        """
        """
        stroke_data= {}
        x = data.x.to(self.device).requires_grad_(self.is_train)
        label = data.y.to(self.device)
        edge_index = data.edge_index.to(self.device)
        stroke_data['stroke_idx'] = data.stroke_idx.to(self.device)
        stroke_data['batch'] = data.batch.to(self.device)
        stroke_data['edge_attr'] = data.edge_attr.to(self.device)
        stroke_data['pos'] = x

        self.optimizer.zero_grad()
        out = self.forward(x, edge_index, stroke_data)
        self.backward(out, label)
        self.optimizer.step()

    def test_time(self, data):
        """
        x: (B*N, F)
        """
        stroke_data= {}
        x = data.x.to(self.device).requires_grad_(self.is_train)
        label = data.y.to(self.device)
        edge_index = data.edge_index.to(self.device)
        stroke_data['stroke_idx'] = data.stroke_idx.to(self.device)
        stroke_data['batch'] = data.batch.to(self.device)
        stroke_data['edge_attr'] = data.edge_attr.to(self.device)
        stroke_data['pos'] = x

        out = self.forward(x, edge_index, stroke_data)
        
        return out
    
    def test(self, data, if_eval=False):
        """
        x: (B*N, F)
        """
        stroke_data= {}
        x = data.x.to(self.device).requires_grad_(self.is_train)
        label = data.y.to(self.device)
        edge_index = data.edge_index.to(self.device)
        stroke_data['stroke_idx'] = data.stroke_idx.to(self.device)
        stroke_data['batch'] = data.batch.to(self.device)
        stroke_data['edge_attr'] = data.edge_attr.to(self.device)
        stroke_data['pos'] = x

        out = self.forward(x, edge_index, stroke_data)
        predict = torch.argmax(out, dim=1).cpu().numpy()

        if (label < 0).any(): # for meaningless label
            self.loss = torch.Tensor([0])
        else:
            self.loss = self.loss_func(out, label)

        return self.loss, predict
        
    
    def print_detail(self):
        print(self.net)

    def update_learning_rate(self):
        """
        update learning rate (called once every epoch)
        """
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def save_network(self, epoch):
        """
        save model to disk
        """
        path = os.path.join(self.save_dir, 
                            '{}_{}.pkl'.format(self.net_name, epoch))
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), path)
    
    def load_network(self, epoch, mode='test'):
        """
        load model from disk
        """
        path = os.path.join(self.save_dir if mode =='test' else self.pretrain_dir, 
                            '{}_{}.pkl'.format(self.net_name, epoch))
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from {}'.format(path))
        state_dict = torch.load(path, map_location=self.device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)
    
