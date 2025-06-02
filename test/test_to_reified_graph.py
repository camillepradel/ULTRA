import torch
import pytest

from torch_geometric.data import HeteroData
from ultra.tasks import ToReifiedGraph


def generate_data() -> HeteroData:
    """
    Returns a hetero data looking like that:
    (a) -[a_to_a]-> (a) -[a_to_b]-> (b)
    """
    data = HeteroData()
    data['a'].num_nodes = 2
    data['a'].feat = torch.rand((2, 10))
    data['b'].num_nodes = 1
    data['a', 'a_to_a', 'a'].edge_index = torch.tensor([[0], [1]])
    data['a', 'a_to_b', 'b'].edge_index = torch.tensor([[1], [0]])
    data['a', 'a_to_b', 'b'].rel_feat = torch.rand((1, 10))
    return data


def test_to_reified_graph():
    assert str(ToReifiedGraph()) == 'ToReifiedGraph()'

    with pytest.raises(ValueError):
        ToReifiedGraph()(HeteroData())

    original_data = generate_data()
    reified_data = ToReifiedGraph()(original_data)
    expected_node_types = [
        "node_instance",
        "node_type",
        "relation_instance",
        "relation_type",
    ]
    assert set(reified_data.node_types) == set(expected_node_types)
    expected_edge_types = [
        ('node_instance', 'has_type', 'node_type'),
        ('node_type', 'has_instance', 'node_instance'),
        ('relation_instance', 'has_type', 'relation_type'),
        ('relation_type', 'has_instance', 'relation_instance'),
        ('relation_instance', 'has_head', 'node_instance'),
        ('node_instance', 'is_head_of', 'relation_instance'),
        ('relation_instance', 'has_tail', 'node_instance'),
        ('node_instance', 'is_tail_of', 'relation_instance'),
    ]
    assert set(reified_data.edge_types) == set(expected_edge_types)
    assert reified_data["node_instance"].num_nodes == 3
    assert reified_data["node_type"].num_nodes == 2
    assert reified_data["relation_instance"].num_nodes == 2
    assert reified_data["relation_type"].num_nodes == 2

    assert (reified_data["node_instance"].feat[0:2] == original_data['a'].feat).all()
    assert reified_data["node_instance"].feat[2].isnan().all()
    assert reified_data["relation_instance"].rel_feat[0].isnan().all()
    assert (reified_data["relation_instance"].rel_feat[1] == original_data['a', 'a_to_b', 'b'].rel_feat).all()

    assert reified_data["node_instance", "has_type", "node_type"].edge_index.tolist() == [
        [0, 1, 2],
        [0, 0, 1],
    ]
    assert reified_data["node_type", "has_instance", "node_instance"].edge_index.tolist() == [
        [0, 0, 1],
        [0, 1, 2],
    ]
    assert reified_data["relation_instance", "has_type", "relation_type"].edge_index.tolist() == [
        [0, 1],
        [0, 1],
    ]
    assert reified_data["relation_type", "has_instance", "relation_instance"].edge_index.tolist() == [
        [0, 1],
        [0, 1],
    ]
    assert reified_data["relation_instance", "has_head", "node_instance"].edge_index.tolist() == [
        [0, 1],
        [0, 1],
    ]
    assert reified_data["node_instance", "is_head_of", "relation_instance"].edge_index.tolist() == [
        [0, 1],
        [0, 1],
    ]
    assert reified_data["relation_instance", "has_tail", "node_instance"].edge_index.tolist() == [
        [0, 1],
        [1, 2],
    ]
    assert reified_data["node_instance", "is_tail_of", "relation_instance"].edge_index.tolist() == [
        [1, 2],
        [0, 1],
    ]
