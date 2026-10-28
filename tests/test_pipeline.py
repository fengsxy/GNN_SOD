import numpy as np
import pytest
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from encoder import FeatureEncoder
from losses import mixed_loss
from Model import GCN
from pre_process import pool_regions, spatial_graph
from runtime import check_config, check_model_version, configure


@pytest.fixture(autouse=True)
def cpu_runtime():
    configure("cpu", threads=2, seed=7)


def test_reverse_map_uses_per_image_offsets_and_keeps_gradients():
    graphs = [Data(x=torch.zeros(2, 1), sp=torch.tensor([[[0, 1], [1, 0]]])),
              Data(x=torch.zeros(3, 1), sp=torch.tensor([[[2, 1], [0, 2]]]))]
    batch = Batch.from_data_list(graphs)
    nodes = torch.tensor([[10.], [11.], [20.], [21.], [22.]], requires_grad=True)
    mapped = GCN.reverse_map(nodes, batch)
    torch.testing.assert_close(mapped[:, 0], torch.tensor([[[10., 11.], [11., 10.]],
                                                          [[22., 21.], [20., 22.]]]))
    mapped.sum().backward()
    torch.testing.assert_close(nodes.grad[:, 0], torch.tensor([2., 2., 1., 1., 2.]))


def test_region_pooling_computes_means_and_gradients():
    feature = torch.tensor([[[[2., 4.], [6., 12.]]]], requires_grad=True)
    labels = torch.tensor([[0, 0], [0, 1]])
    pooled = pool_regions(feature, labels)
    torch.testing.assert_close(pooled[:, 0], torch.tensor([4., 12.]))
    pooled.sum().backward()
    torch.testing.assert_close(feature.grad, torch.tensor([[[[1/3, 1/3], [1/3, 1.]]]]))


def make_sample(node_counts):
    graphs = []
    for count in node_counts:
        graph = Data(x=torch.rand(count, 1536),
                     edge_index=torch.stack((torch.arange(count), torch.arange(count).roll(1))),
                     edge_attr=torch.ones(count),
                     sp=(torch.arange(32 * 32) % count).reshape(1, 32, 32))
        graphs.append(graph)
    for i, (channels, size) in enumerate(((512, 2), (512, 4), (256, 8), (128, 16))):
        setattr(graphs[0], f"cnn_{i}", torch.rand(1, channels, size, size))
    graphs[0].y = torch.randint(2, (1, 1, 32, 32)).float()
    return graphs


@pytest.mark.parametrize("batch_size", [1, 2])
def test_every_branch_trains_with_variable_node_counts(batch_size):
    samples = [make_sample((5, 7)), make_sample((8, 11))][:batch_size]
    batch = next(iter(DataLoader(samples, batch_size=batch_size)))
    model = GCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    before = [branch[0].linB.weight.detach().clone() for branch in model.graph_layers]
    outputs = model(batch)
    assert len(outputs) == 5
    assert all(t.shape == (batch_size, 1, 32, 32) for t in outputs)
    loss = sum(mixed_loss(t, batch[0].y) for t in outputs)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    optimizer.step()
    for old, branch in zip(before, model.graph_layers):
        assert not torch.equal(old, branch[0].linB.weight)


@pytest.mark.parametrize("value", [0., 1.])
def test_loss_handles_empty_and_full_masks(value):
    prediction = torch.full((2, 1, 16, 16), .5, requires_grad=True)
    loss = mixed_loss(prediction, torch.full_like(prediction, value))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    with pytest.raises(ValueError, match="matching"):
        mixed_loss(prediction[:, 0], prediction[:, 0])


def test_encoder_shapes_checkpoint_and_weight_identity(tmp_path):
    encoder = FeatureEncoder(weights="none", seed=0)
    high, skips = encoder(torch.rand(1, 3, 64, 64))
    assert [tuple(f.shape) for f in high] == [(1, 512, 8, 8), (1, 512, 4, 4), (1, 512, 2, 2)]
    assert [tuple(f.shape) for f in skips] == [(1, 512, 4, 4), (1, 512, 8, 8),
                                             (1, 256, 16, 16), (1, 128, 32, 32)]
    signature = encoder.fingerprint()
    torch.save({"encoder_state": encoder.state_dict()}, tmp_path / "encoder.pt")
    restored = FeatureEncoder(weights="imagenet", checkpoint=tmp_path / "encoder.pt")
    assert restored.fingerprint() == signature  # No pretrained download when a checkpoint is given.
    assert all(not p.requires_grad for p in restored.parameters())
    with torch.no_grad():
        restored.features[0].weight[0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="feature_signature"):
        check_config({"feature_signature": signature}, {"feature_signature": restored.fingerprint()})


def test_graph_retains_legacy_sparsification_without_self_loops():
    coord = np.array([[0., 0.], [.1, .2], [.6, .5], [.8, .9]])
    edges, weights = spatial_graph(coord, 2)
    assert (edges[0] != edges[1]).all()
    assert torch.bincount(edges[0], minlength=4).max() <= 2
    assert torch.bincount(edges[1], minlength=4).max() <= 2
    distances = np.linalg.norm(coord[edges[0]] - coord[edges[1]], axis=1)
    np.testing.assert_allclose(weights.numpy(), np.exp(-distances / (.1 * np.pi) ** 2), rtol=1e-6)


def test_later_stages_receive_the_immediately_previous_fusion():
    model = GCN().eval()
    batch = next(iter(DataLoader([make_sample((5, 7))], batch_size=1)))
    observed = []
    handles = []
    # Intervene on each fused feature with a different marker. Later branches
    # must receive their direct predecessor, not the first stage repeatedly.
    for stage, layer in enumerate(model.scale_fusion):
        handles.append(layer.register_forward_hook(
            lambda module, args, output, value=stage + 1: torch.full_like(output, value)))
    for branch in model.branch_fusion:
        for stage in range(1, 4):
            width, previous = (64, 32, 16, 8)[stage], (32, 16, 8, 4)[stage - 1]
            def capture(module, args, w=width, p=previous, expected=stage):
                torch.testing.assert_close(args[0][:, w:w+p],
                                           torch.full_like(args[0][:, w:w+p], expected))
                observed.append(expected)
            handles.append(branch[stage].register_forward_pre_hook(capture))
    with torch.no_grad():
        model(batch)
    assert sorted(observed) == [1, 1, 2, 2, 3, 3]
    for handle in handles:
        handle.remove()


def test_rejects_checkpoints_from_the_first_fusion_implementation():
    check_model_version({"format_version": 2})
    with pytest.raises(ValueError, match="sequential fusion"):
        check_model_version({"format_version": 1})
