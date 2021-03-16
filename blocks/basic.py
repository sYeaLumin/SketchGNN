import torch.nn as nn

def act_layer(act_type, inplace=False, neg_slope=0.2, n_prelu=1):
    """
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return layer


def norm_layer(norm_type, nc):
    """
    """
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm1d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm1d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return layer


class MLPConv(nn.Sequential):
    def __init__(self, channels, act_type='relu', norm_type='batch', bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=bias))
            if norm_type:
                m.append(norm_layer(norm_type, channels[i]))
            if act_type:
                m.append(act_layer(act_type))
        super(MLPConv, self).__init__(*m)
    
    def forward(self, x):
        """
        x:B x N x F
        """
        x = x.transpose(1, 2) 
        for module in self._modules.values():
            x = module(x)
        x = x.transpose(1, 2) 
        return x


class MLPLinear(nn.Sequential):
    def __init__(self, channels, act_type='relu', norm_type='batch', bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))
            if norm_type and norm_type != 'None':
                m.append(norm_layer(norm_type, channels[i]))
            if act_type:
                m.append(act_layer(act_type))
        super(MLPLinear, self).__init__(*m)


class MultiSeq(nn.Sequential):
    def __init__(self, *args):
        super(MultiSeq, self).__init__(*args)

    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs



