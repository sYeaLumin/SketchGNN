import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_scatter

# from .layers import GATLayer, PoolLayer
from .basic import MLPLinear
from .dynamic import DilatedKnnGraph
from .tg_conv import MRConv, EdgeConv2, SAGEConv2

def cal_edge_attr(edge_index, pos):
    row = edge_index[0]
    col = edge_index[1]
    offset = pos[col] - pos[row]
    dist = torch.norm(offset, p=2, dim=-1).view(-1, 1)
    theta = torch.atan2(offset[:,1], offset[:,0]).view(-1, 1)
    edge_attr = torch.cat([offset, dist, theta], dim=-1)
    return edge_attr

class GraphConv(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, opt):
        super(GraphConv, self).__init__()
        self.gcn_type = opt.gcn_type
        if self.gcn_type == 'edge':
            self.gconv = tg.nn.EdgeConv(
                nn=MLPLinear(
                    channels=[in_channels*2, out_channels],
                    act_type=opt.act_type, 
                    norm_type=opt.norm_type
                ),
                aggr=opt.aggr_type
            )
        elif self.gcn_type == 'edge2':
            self.gconv = EdgeConv2(
                nn=MLPLinear(
                    channels=[in_channels*2, out_channels], 
                    act_type=opt.act_type, 
                    norm_type=None
                ),
                aggr=opt.aggr_type
            )
        elif self.gcn_type == 'mr':
            self.gconv = MRConv(
                nn=MLPLinear(
                    channels=[in_channels*2, out_channels],
                    act_type=opt.act_type, 
                    norm_type=None
                ),
                aggr=opt.aggr_type
            )
        elif self.gcn_type == 'sage':
            self.gconv = tg.nn.SAGEConv(
                in_channels=in_channels,
                out_channels=out_channels,
                normalize=False,
                # concat=False, 
                bias=True
            )
        elif self.gcn_type == 'sage2':
            self.gconv = SAGEConv2(
                local_nn=MLPLinear(
                    channels=[in_channels, in_channels],
                    act_type=opt.act_type, 
                    norm_type=None
                ),
                global_nn=MLPLinear(
                    channels=[in_channels*2, out_channels],
                    act_type=opt.act_type, 
                    norm_type=None
                ),
                aggr=opt.aggr_type
            )
        elif self.gcn_type == 'gin':
            self.gconv = tg.nn.GINConv(
                nn=MLPLinear(
                    channels=[in_channels, out_channels],
                    act_type=opt.act_type, 
                    norm_type=None
                ),
                eps=0,
                train_eps=False
            )
        elif self.gcn_type == 'gin+':
            self.gconv = tg.nn.GINConv(
                nn=MLPLinear(
                    channels=[in_channels, out_channels],
                    act_type=opt.act_type, 
                    norm_type=None
                ),
                eps=0.1,
                train_eps=True
            )
        elif self.gcn_type == 'ecc':
            self.gconv = tg.nn.NNConv(
                in_channels=in_channels,
                out_channels=out_channels,
                nn=nn.Linear(4, in_channels*out_channels, bias=False),
                aggr='mean',
                root_weight=False,
                bias=True
            )            
        else:
            raise NotImplementedError('conv_type {} is not implemented. Please check.\n'.format(opt.conv_type))

    def forward(self, x, edge_index, data=None):
        """
        x: (BxN) x F
        """
        if self.gcn_type == 'ecc':
            return self.gconv(x, edge_index, data['edge_attr'])
        else:
            return self.gconv(x, edge_index)

class DynConv(GraphConv):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, dilation, opt, knn_type='matrix'):
        super(DynConv, self).__init__(in_channels, out_channels, opt)
        self.k = opt.kernel_size
        self.d = dilation
        self.dilated_knn_graph = DilatedKnnGraph(opt.kernel_size, dilation, opt.stochastic, opt.epsilon, knn_type=knn_type)
        self.mixedge = opt.mixedge

    def forward(self, x, edge_index, data=None):
        """
        x: (BxN) x F
        """
        dyn_edge_index = self.dilated_knn_graph(x, data['batch'])
        if self.mixedge:
            dyn_edge_index = torch.unique(torch.cat([edge_index, dyn_edge_index], dim=1), dim=1)
        
        # TODO: calculate edge_attr use pos
        if self.gcn_type == 'ecc':
            dyn_edge_attr = cal_edge_attr(dyn_edge_index, data['pos'])
        else:
            dyn_edge_attr = None

        return super(DynConv, self).forward(x, dyn_edge_index, {'edge_attr':dyn_edge_attr})

class ResGcnBlock(nn.Module):
    """
    Residual Dynamic graph convolution block
    """
    def __init__(self, channels, adj_type, dilation, opt):
        super(ResGcnBlock, self).__init__()
        if adj_type == 'static':
            self.body = GraphConv(channels, channels, opt)
        elif adj_type == 'dynamic':
            self.body = DynConv(channels, channels, dilation, opt)
        else:
            raise NotImplementedError('adj_type {} is not implemented. Please check.\n'.format(opt.adj_type))

    # input: (x0, x1, x2, ..., xi);  (xi-1, xi), output is (xi, xi+1)
    def forward(self, x, edge_index, data=None):
        """
        x: (BxN) x F
        edge_index: 2xE
        """
        res = self.body(x[:,:,-1], edge_index, data)
        res = res + x[:,:,-1] # res (BxN) x F
        return torch.cat((x, res.unsqueeze(-1)), 2), edge_index, data

class MixPool(nn.Module):
    """
    Dense Dynamic graph convolution block
    """
    def __init__(self, in_channel, out_channel):
        super(MixPool, self).__init__()
        self.trans_max = MLPLinear([in_channel, out_channel],
                                norm_type='batch', act_type='relu')
        self.trans_sketch = MLPLinear([in_channel, out_channel],
                                norm_type='batch', act_type='relu')

    def forward(self, x, stroke_idx, batch=None):
        """
        x: (BxN) x F
        stroke_idx:2xEE
        """
        pool_max = self.trans_max(x)
        pool_sketch = self.trans_sketch(x)
        F = pool_max.shape[1]
        pool_max = torch.index_select(tg.nn.global_max_pool(pool_max, batch), 0, batch)
        # pool_sketch = tg.utils.scatter_("max", torch.index_select(pool_sketch, 0, pool_edge_index[0]), pool_edge_index[1])
        # pool_sketch = torch_scatter.scatter_max(torch.index_select(pool_sketch, 0, pool_edge_index[0]), pool_edge_index[1], 0)[0]
        pool_sketch = torch.index_select(tg.nn.global_max_pool(pool_sketch, stroke_idx), 0, stroke_idx)
        return torch.cat([pool_sketch, pool_max], dim=1)


class MaxPool(nn.Module):
    """
    Dense Dynamic graph convolution block
    """
    def __init__(self, in_channel, out_channel):
        super(MaxPool, self).__init__()
        self.trans_max = MLPLinear([in_channel, out_channel],
                                norm_type='batch', act_type='relu')

    def forward(self, x, stroke_idx, batch=None):
        """
        x: (BxN) x F
        pool_edge_index:2xEE
        """
        pool_max = self.trans_max(x)
        pool_max = torch.index_select(tg.nn.global_max_pool(pool_max, batch), 0, batch)
        return pool_max
