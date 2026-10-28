"""Multi-scale graph decoder. Inputs and all five outputs retain a batch axis."""
import torch
from torch import nn
from torch.nn import functional as F
from Module import GatedEdgeConv, GraphSizeNorm


def conv_block(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                         nn.BatchNorm2d(out_channels), nn.ReLU())


class GCN(nn.Module):
    def __init__(self, in_features=1536, n_scales=2):
        super().__init__()
        if n_scales < 1:
            raise ValueError("At least one superpixel scale is required")
        self.in_features = in_features
        self.n_scales = n_scales
        widths, fused = (64, 32, 16, 8), (32, 16, 8, 4)
        inputs = (in_features,) + widths[:-1]
        self.graph_layers = nn.ModuleList([
            nn.ModuleList([GatedEdgeConv(i, o) for i, o in zip(inputs, widths)])
            for _ in range(n_scales)])
        self.graph_bn = nn.ModuleList([
            nn.ModuleList([nn.BatchNorm1d(w) for w in widths]) for _ in range(n_scales)])
        self.graph_norm = GraphSizeNorm()
        self.cnn_reduce = nn.ModuleList([
            nn.Conv2d(c, w, 1) for c, w in zip((512, 512, 256, 128), widths)])
        self.reverse_reduce = nn.ModuleList([
            nn.ModuleList([conv_block(w, w) for w in widths]) for _ in range(n_scales)])
        self.branch_fusion = nn.ModuleList([
            nn.ModuleList([conv_block(2 * w + (fused[stage - 1] if stage else 0), w)
                           for stage, w in enumerate(widths)]) for _ in range(n_scales)])
        self.scale_fusion = nn.ModuleList([conv_block(w, f) for w, f in zip(widths, fused)])
        self.side_heads = nn.ModuleList([nn.Conv2d(f, 1, 1) for f in fused])
        self.final_head = nn.Sequential(conv_block(sum(fused), sum(fused)),
                                        nn.Conv2d(sum(fused), 1, 1))

    @staticmethod
    def reverse_map(nodes, graph):
        """Gather local region features, using PyG pointers for each image.

        Indexing preserves autograd; node counts need not be equal across images.
        """
        maps = []
        for i in range(graph.num_graphs):
            local = nodes[graph.ptr[i]:graph.ptr[i + 1]]
            labels = graph.sp[i]
            if labels.min() < 0 or labels.max() >= local.shape[0]:
                raise ValueError("Superpixel labels do not index this image's nodes")
            maps.append(local[labels].permute(2, 0, 1))
        return torch.stack(maps)

    def forward(self, data_list, batch_list=None):
        if len(data_list) != self.n_scales:
            raise ValueError(f"Expected {self.n_scales} graph scales, got {len(data_list)}")
        size = data_list[0].sp.shape[-2:]
        for graph in data_list:
            if graph.x.shape[1] != self.in_features or graph.sp.shape[-2:] != size:
                raise ValueError("Graph features or image sizes do not match the model")
        cnn = [layer(F.interpolate(getattr(data_list[0], f"cnn_{i}"), size=size,
                                   mode="bilinear", align_corners=False))
               for i, layer in enumerate(self.cnn_reduce)]
        nodes = [g.x for g in data_list]
        fused_maps, outputs = [], []
        for stage in range(4):
            branch_maps = []
            for scale, graph in enumerate(data_list):
                nodes[scale] = F.relu(self.graph_bn[scale][stage](self.graph_norm(
                    self.graph_layers[scale][stage](nodes[scale], graph.edge_index,
                                                    graph.edge_attr), graph.batch)))
                mapped = self.reverse_reduce[scale][stage](self.reverse_map(nodes[scale], graph))
                # Manuscript Fig. 2 and Eq. (6): feed the immediately previous fused feature.
                parts = [mapped] + ([fused_maps[-1]] if stage else []) + [cnn[stage]]
                branch_maps.append(self.branch_fusion[scale][stage](torch.cat(parts, dim=1)))
            fused = self.scale_fusion[stage](torch.stack(branch_maps).sum(dim=0))
            fused_maps.append(fused)
            outputs.append(torch.sigmoid(self.side_heads[stage](fused)))
        outputs.append(torch.sigmoid(self.final_head(torch.cat(fused_maps, dim=1))))
        return tuple(outputs)
