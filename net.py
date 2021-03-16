import torch
import torch.nn as nn
from blocks.conv import *
from blocks.basic import MLPLinear, MultiSeq
import torch_geometric.nn as tgnn
# import torchsnooper


def init_net(opt):
    if opt.net_name == 'SketchOneLine':
        net = SketchOneLine(opt)
    elif opt.net_name == 'SketchTwoLine':
        net = SketchTwoLine(opt)
    else:
        raise NotImplementedError('net {} is not implemented. Please check.\n'.format(opt.net_name))
    
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(opt.gpu_ids[0])
        net = net.cuda()
        net = torch.nn.DataParallel(net, device_ids=opt.gpu_ids)
    return net

class SketchOneLine(nn.Module):
    def __init__(self, opt):
        super(SketchOneLine, self).__init__()
        self.opt = opt
        self.n_blocks = opt.n_blocks
        self.channels = opt.channels
        self.pool_channels = opt.pool_channels
        dilations = [1, 4, 8] + [opt.global_dilation] * (self.n_blocks-2)
            
        # head
        if opt.adj_type == 'static':
            self.head = GraphConv(opt.in_feature, self.channels, opt)
        elif opt.adj_type == 'dynamic':
            self.head = DynConv(opt.in_feature, self.channels, dilations[0], opt)
        else:
            raise NotImplementedError('adj_type {} is not implemented. Please check.\n'.format(opt.adj_type))
        
        # backbone
        self.backbone = MultiSeq(*[ResGcnBlock(self.channels, opt.adj_type, dilations[i+1], opt) for i in range(self.n_blocks)])
        
        if opt.fusion_type == 'mix':
            self.pool = MixPool(opt.channels*(opt.n_blocks+1), opt.pool_channels // 2)
            mlpSegment = [self.channels*(self.n_blocks+1) + self.pool_channels] + opt.mlp_segment
        elif opt.fusion_type == 'max':
            self.pool = MaxPool(opt.channels*(opt.n_blocks+1), opt.pool_channels)
            mlpSegment = [self.channels*(self.n_blocks+1) + self.pool_channels] + opt.mlp_segment
        else:
            raise NotImplementedError('fusion_type {} is not implemented. Please check.\n'.format(opt.fusion_type))
        self.segment = MultiSeq(*[MLPLinear(mlpSegment, norm_type='batch', act_type='relu'),
                                  MLPLinear([mlpSegment[-1], opt.out_segment], norm_type='batch', act_type=None)])
        
        # softmax        
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        
    # @torchsnooper.snoop()
    def forward(self, x, edge_index, data):
        """
        x: (BxN) x F
        """
        BN = x.shape[0]
        x = self.head(x, edge_index, data).unsqueeze(-1)
        x = torch.cat((x, x), 2)
        x = self.backbone(x, edge_index, data)[0][:,:,1:].contiguous().view(BN, -1)
        x_g = self.pool(x, data['stroke_idx'], data['batch'])

        ####################### segment #######################
        x = torch.cat([x, x_g], dim=1)
        x = self.segment(x)
        return self.LogSoftmax(x)


class SketchTwoLine(nn.Module):
    def __init__(self, opt):
        super(SketchTwoLine, self).__init__()
        self.opt = opt
        self.n_blocks = opt.n_blocks
        self.channels = opt.channels
        self.pool_channels = opt.pool_channels

        ####################### point feature #######################
        opt.kernel_size = opt.local_k
        opt.dilation = opt.local_dilation
        opt.stochastic = opt.local_stochastic
        opt.epsilon = opt.local_epsilon
        dilations = [1, 4, 8] + [opt.local_dilation] * (self.n_blocks-2)   
        
        # head
        if self.opt.local_adj_type == 'static':
            self.local_head = GraphConv(opt.in_feature, self.channels, opt)
        else:
            self.local_head = DynConv(opt.in_feature, self.channels, dilations[0], opt) 
        
        # local backbone
        self.local_backbone = MultiSeq(*[ResGcnBlock(self.channels, opt.local_adj_type, dilations[i+1], opt) for i in range(self.n_blocks)])

        ####################### stroke & sketch feature #######################
        opt.kernel_size = opt.global_k
        opt.dilation = opt.global_dilation
        opt.stochastic = opt.global_stochastic
        opt.epsilon = opt.global_epsilon
        dilations = [1, opt.global_dilation//4, opt.global_dilation//2] + [opt.global_dilation] * (self.n_blocks-2)   
        
        # head
        if self.opt.global_adj_type == 'static':
            self.global_head = GraphConv(opt.in_feature, self.channels, opt)
        else:
            self.global_head = DynConv(opt.in_feature, self.channels, dilations[0], opt)    
        
        # global backbone
        self.global_backbone = MultiSeq(*[ResGcnBlock(self.channels, opt.global_adj_type, dilations[i+1], opt) for i in range(self.n_blocks)])
        
        if opt.fusion_type == 'mix':
            self.pool = MixPool(opt.channels*(opt.n_blocks+1), opt.pool_channels // 2)
            mlpSegment = [self.channels*(self.n_blocks+1) + self.pool_channels] + opt.mlp_segment
        elif opt.fusion_type == 'max':
            self.pool = MaxPool(opt.channels*(opt.n_blocks+1), opt.pool_channels)
            mlpSegment = [self.channels*(self.n_blocks+1) + self.pool_channels] + opt.mlp_segment
        else:
            raise NotImplementedError('fusion_type {} is not implemented. Please check.\n'.format(opt.fusion_type))
        self.segment = MultiSeq(*[MLPLinear(mlpSegment, norm_type='batch', act_type='relu'),
                                  MLPLinear([mlpSegment[-1], opt.out_segment], norm_type='batch', act_type=None)])
        # softmax        
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        
    # @torchsnooper.snoop()
    def forward(self, x, edge_index, data):
        """
        x: (BxN) x F
        """
        BN = x.shape[0]
        ####################### local line #######################
        x_l = self.local_head(x, edge_index, data).unsqueeze(-1)
        x_l = torch.cat((x_l, x_l), 2)
        x_l = self.local_backbone(x_l, edge_index, data)[0][:,:,1:].contiguous().view(BN, -1)

        ####################### global line #######################
        x_g = self.global_head(x, edge_index, data).unsqueeze(-1)
        x_g = torch.cat((x_g, x_g), 2)
        x_g = self.global_backbone(x_g, edge_index, data)[0][:,:,1:].contiguous().view(BN, -1)
        x_g = self.pool(x_g, data['stroke_idx'], data['batch'])

        ####################### segment #######################
        x = torch.cat([x_l, x_g], dim=1)
        x = self.segment(x)
        return self.LogSoftmax(x)

    
if __name__ == "__main__":
    import os
    import ndjson
    from options import TrainOptions
    from utils import load_data  
    _opt = TrainOptions().parse()
    model = init_net(_opt)
    print(model)