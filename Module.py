from torch_geometric.utils import degree
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn.conv import MessagePassing


class GatedEdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GatedEdgeConv, self).__init__(aggr='add')  # "Max" aggregation. ['add', 'mean', 'max']
        self.linB = torch.nn.Linear(in_channels, out_channels)
        self.linA = torch.nn.Linear(in_channels, out_channels)
        self.mlp = Seq(Linear(out_channels, 2 * out_channels),
                       ReLU(),
                       Linear(2 * out_channels, out_channels))
        self.sigma = 0.1 * 3.1415926  # 3.1415926*0.1

    def forward(self, x0, edge_index, edge_attr_dist):
        # x0 has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_index, _ = remove_self_loops(edge_index)
        x = self.linB(x0)
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, x0=x0, edge_attr_dist=edge_attr_dist)

    def message(self, x_i, x_j, edge_attr_dist):  # massage flow from node_j to node_i
        # # x_i has shape [E, in_channels]
        # # x_j has shape [E, in_channels]
        # # edge_attr_dist has shape E, we reshape it as [E, 1]
        edge_attr_dist = edge_attr_dist.view(-1, 1)
        # dist = torch.log(edge_attr_dist)*(-self.sigma ** 2)
        # distNorm = dist / torch.max(dist)
        # tmp = torch.cat([(x_j-x_i) * distNorm * 10], dim=1)  # tmp has shape [E, in_channels]
        # tmp_g = (self.mlp(tmp).abs() + 0.000001).pow(-1.0)
        # gate = (2 * torch.sigmoid(tmp_g) - 1)
        tmp = torch.cat([(x_j-x_i) * edge_attr_dist], dim=1)  # tmp has shape [E, in_channels]
        gate = self.mlp(tmp)
        return gate * x_j


    def update(self, aggr_out, x0):
        # aggr_out has shape [N, out_channels]

        return aggr_out + self.linA(x0)

class BatchNorm(torch.nn.BatchNorm1d):
    r"""Applies batch normalization over a batch of node features as described
    in the `"Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariate Shift" <https://arxiv.org/abs/1502.03167>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} -
        \textrm{E}[\mathbf{x}]}{\sqrt{\textrm{Var}[\mathbf{x}] + \epsilon}}
        \odot \gamma + \beta

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        momentum (float, optional): The value used for the running mean and
            running variance computation. (default: :obj:`0.1`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
        track_running_stats (bool, optional): If set to :obj:`True`, this
            module tracks the running mean and variance, and when set to
            :obj:`False`, this module does not track such statistics and always
            uses batch statistics in both training and eval modes.
            (default: :obj:`True`)
    """

    def __init__(self, in_channels, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm, self).__init__(in_channels, eps, momentum, affine,
                                        track_running_stats)

    def forward(self, x):
        """"""
        return super(BatchNorm, self).forward(x)

    def __repr__(self):
        return ('{}({}, eps={}, momentum={}, affine={}, '
                'track_running_stats={})').format(self.__class__.__name__,
                                                  self.num_features, self.eps,
                                                  self.momentum, self.affine,
                                                  self.track_running_stats)


class GraphSizeNorm(nn.Module):
    r"""Applies Graph Size Normalization over each individual graph in a batch
    of node features as described in the
    `"Benchmarking Graph Neural Networks" <https://arxiv.org/abs/2003.00982>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x}_i}{\sqrt{|\mathcal{V}|}}
    """

    def __init__(self):
        super(GraphSizeNorm, self).__init__()

    def forward(self, x, batch=None):
        """"""
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        inv_sqrt_deg = degree(batch, dtype=x.dtype).pow(-0.5)
        return x * inv_sqrt_deg[batch].view(-1, 1)