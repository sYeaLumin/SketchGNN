import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset

try:
    from torch_cluster import knn_graph
except ImportError:
    knn_graph = None

class MRConv(MessagePassing):
    def __init__(self, nn, aggr='max', **kwargs):
        super(MRConv, self).__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.reset_parameters()
    
    def reset_parameters(self):
        reset(self.nn)  
    
    def forward(self, x, edge_index):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return x_j - x_i

    def update(self, aggr_out, x):
        return self.nn(torch.cat([x, aggr_out], dim=-1))

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class EdgeConv2(MessagePassing):
    def __init__(self, nn, aggr='max', **kwargs):
        super(EdgeConv2, self).__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.reset_parameters()
    
    def reset_parameters(self):
        reset(self.nn)
    
    
    def forward(self, x, edge_index):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return self.nn(torch.cat([x_i, x_j], dim=1))

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class SAGEConv2(MessagePassing):
    def __init__(self, local_nn, global_nn, aggr='max', **kwargs):
        super(SAGEConv2, self).__init__(aggr=aggr, **kwargs)
        self.local_nn = local_nn
        self.global_nn = global_nn
        self.reset_parameters()
    
    def reset_parameters(self):
        reset(self.local_nn)
        reset(self.global_nn) 
    
    def forward(self, x, edge_index):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j):
        return self.local_nn(x_j)
    
    def update(self, aggr_out, x):
        return self.global_nn(torch.cat([x, aggr_out], dim=-1))

    def __repr__(self):
        return '{}(local_nn={},gloabl_nn={})'.format(self.__class__.__name__, self.local_nn, self.global_nn)