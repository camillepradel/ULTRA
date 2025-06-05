import copy
from functools import reduce
from torch_scatter import scatter_add
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
import torch
from typing import Any
from torch import Tensor
from torch_geometric.typing import NodeType
from torch_geometric.utils import bipartite_subgraph

from typing import Dict


def edge_match(edge_index, query_index):
    # O((n + q)logn) time
    # O(n) memory
    # edge_index: big underlying graph
    # query_index: edges to match

    # preparing unique hashing of edges, base: (max_node, max_relation) + 1
    base = edge_index.max(dim=1)[0] + 1
    # we will map edges to long ints, so we need to make sure the maximum product is less than MAX_LONG_INT
    # idea: max number of edges = num_nodes * num_relations
    # e.g. for a graph of 10 nodes / 5 relations, edge IDs 0...9 mean all possible outgoing edge types from node 0
    # given a tuple (h, r), we will search for all other existing edges starting from head h
    assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max
    scale = base.cumprod(0)
    scale = scale[-1] // scale

    # hash both the original edge index and the query index to unique integers
    edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)
    edge_hash, order = edge_hash.sort()
    query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)

    # matched ranges: [start[i], end[i])
    start = torch.bucketize(query_hash, edge_hash)
    end = torch.bucketize(query_hash, edge_hash, right=True)
    # num_match shows how many edges satisfy the (h, r) pattern for each query in the batch
    num_match = end - start

    # generate the corresponding ranges
    offset = num_match.cumsum(0) - num_match
    range = torch.arange(num_match.sum(), device=edge_index.device)
    range = range + (start - offset).repeat_interleave(num_match)

    return order[range], num_match


def negative_sampling(data, batch, num_negative, strict=True):
    batch_size = batch.batch_size
    assert batch_size == batch["node_instance"]["num_sampled_nodes"][0]//2
    assert batch_size == batch["relation_type"]["num_sampled_nodes"][0]
    
    pos_h_index = batch["node_instance"]["n_id"][:batch_size]
    pos_t_index = batch["node_instance"]["n_id"][batch_size:batch_size*2]
    pos_r_index = batch["relation_type"]["n_id"][:batch_size]

    # strict negative sampling vs random negative sampling
    if strict:
        # TODO: adapt strict sampling to reified graphs
        t_mask, h_mask = strict_negative_mask(data, batch)
        
        t_mask = t_mask[:batch_size // 2]
        neg_t_candidate = t_mask.nonzero()[:, 1]
        num_t_candidate = t_mask.sum(dim=-1)
        # draw samples for negative tails
        rand = torch.rand(len(t_mask), num_negative, device=batch.device)
        index = (rand * num_t_candidate.unsqueeze(-1)).long()
        index = index + (num_t_candidate.cumsum(0) - num_t_candidate).unsqueeze(-1)
        neg_t_index = neg_t_candidate[index]

        h_mask = h_mask[batch_size // 2:]
        neg_h_candidate = h_mask.nonzero()[:, 1]
        num_h_candidate = h_mask.sum(dim=-1)
        # draw samples for negative heads
        rand = torch.rand(len(h_mask), num_negative, device=batch.device)
        index = (rand * num_h_candidate.unsqueeze(-1)).long()
        index = index + (num_h_candidate.cumsum(0) - num_h_candidate).unsqueeze(-1)
        neg_h_index = neg_h_candidate[index]
    else:
        neg_index = torch.randint(data.num_nodes, (batch_size, num_negative), device=batch["node_instance"]["n_id"].device)
        neg_t_index, neg_h_index = neg_index[:batch_size // 2], neg_index[batch_size // 2:]

    h_index = pos_h_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index = pos_t_index.unsqueeze(-1).repeat(1, num_negative + 1)
    r_index = pos_r_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index[:batch_size // 2, 1:] = neg_t_index
    h_index[batch_size // 2:, 1:] = neg_h_index

    return torch.stack([h_index, t_index, r_index], dim=-1)


def all_negative(data, batch):
    pos_h_index, pos_t_index, pos_r_index = batch.t()
    r_index = pos_r_index.unsqueeze(-1).expand(-1, data["node_instance"].num_nodes)
    # generate all negative tails for this batch
    all_index = torch.arange(data["node_instance"].num_nodes, device=batch.device)
    h_index, t_index = torch.meshgrid(pos_h_index, all_index, indexing="ij")  # indexing "xy" would return transposed
    t_batch = torch.stack([h_index, t_index, r_index], dim=-1)
    # generate all negative heads for this batch
    all_index = torch.arange(data["node_instance"].num_nodes, device=batch.device)
    t_index, h_index = torch.meshgrid(pos_t_index, all_index, indexing="ij")
    h_batch = torch.stack([h_index, t_index, r_index], dim=-1)

    return t_batch, h_batch


def strict_negative_mask(data, batch):
    # this function makes sure that for a given (h, r) batch we will NOT sample true tails as random negatives
    # similarly, for a given (t, r) we will NOT sample existing true heads as random negatives

    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # part I: sample hard negative tails
    # edge index of all (head, relation) edges from the underlying graph
    edge_index = torch.stack([data.edge_index[0], data.edge_type])
    # edge index of current batch (head, relation) for which we will sample negatives
    query_index = torch.stack([pos_h_index, pos_r_index])
    # search for all true tails for the given (h, r) batch
    edge_id, num_t_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    t_truth_index = data.edge_index[1, edge_id]
    sample_id = torch.arange(len(num_t_truth), device=batch.device).repeat_interleave(num_t_truth)
    t_mask = torch.ones(len(num_t_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true tails
    t_mask[sample_id, t_truth_index] = 0
    t_mask.scatter_(1, pos_t_index.unsqueeze(-1), 0)

    # part II: sample hard negative heads
    # edge_index[1] denotes tails, so the edge index becomes (t, r)
    edge_index = torch.stack([data.edge_index[1], data.edge_type])
    # edge index of current batch (tail, relation) for which we will sample heads
    query_index = torch.stack([pos_t_index, pos_r_index])
    # search for all true heads for the given (t, r) batch
    edge_id, num_h_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    h_truth_index = data.edge_index[0, edge_id]
    sample_id = torch.arange(len(num_h_truth), device=batch.device).repeat_interleave(num_h_truth)
    h_mask = torch.ones(len(num_h_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true heads
    h_mask[sample_id, h_truth_index] = 0
    h_mask.scatter_(1, pos_h_index.unsqueeze(-1), 0)

    return t_mask, h_mask


def compute_ranking(pred, target, mask=None):
    pos_pred = pred.gather(-1, target.unsqueeze(-1))
    if mask is not None:
        # filtered ranking
        ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
    else:
        # unfiltered ranking
        ranking = torch.sum(pos_pred <= pred, dim=-1) + 1
    return ranking


def build_relation_graph(graph):

    # expect the graph is already with inverse edges

    edge_index, edge_type = graph.edge_index, graph.edge_type
    num_nodes, num_rels = graph.num_nodes, graph.num_relations
    device = edge_index.device

    Eh = torch.vstack([edge_index[0], edge_type]).T.unique(dim=0)  # (num_edges, 2)
    Dh = scatter_add(torch.ones_like(Eh[:, 1]), Eh[:, 0])

    EhT = torch.sparse_coo_tensor(
        torch.flip(Eh, dims=[1]).T, 
        torch.ones(Eh.shape[0], device=device) / Dh[Eh[:, 0]], 
        (num_rels, num_nodes)
    )
    Eh = torch.sparse_coo_tensor(
        Eh.T, 
        torch.ones(Eh.shape[0], device=device), 
        (num_nodes, num_rels)
    )
    Et = torch.vstack([edge_index[1], edge_type]).T.unique(dim=0)  # (num_edges, 2)

    Dt = scatter_add(torch.ones_like(Et[:, 1]), Et[:, 0])
    assert not (Dt[Et[:, 0]] == 0).any()

    EtT = torch.sparse_coo_tensor(
        torch.flip(Et, dims=[1]).T, 
        torch.ones(Et.shape[0], device=device) / Dt[Et[:, 0]], 
        (num_rels, num_nodes)
    )
    Et = torch.sparse_coo_tensor(
        Et.T, 
        torch.ones(Et.shape[0], device=device), 
        (num_nodes, num_rels)
    )

    Ahh = torch.sparse.mm(EhT, Eh).coalesce()
    Att = torch.sparse.mm(EtT, Et).coalesce()
    Aht = torch.sparse.mm(EhT, Et).coalesce()
    Ath = torch.sparse.mm(EtT, Eh).coalesce()

    hh_edges = torch.cat([Ahh.indices().T, torch.zeros(Ahh.indices().T.shape[0], 1, dtype=torch.long).fill_(0)], dim=1)  # head to head
    tt_edges = torch.cat([Att.indices().T, torch.zeros(Att.indices().T.shape[0], 1, dtype=torch.long).fill_(1)], dim=1)  # tail to tail
    ht_edges = torch.cat([Aht.indices().T, torch.zeros(Aht.indices().T.shape[0], 1, dtype=torch.long).fill_(2)], dim=1)  # head to tail
    th_edges = torch.cat([Ath.indices().T, torch.zeros(Ath.indices().T.shape[0], 1, dtype=torch.long).fill_(3)], dim=1)  # tail to head
    
    rel_graph = Data(
        edge_index=torch.cat([hh_edges[:, [0, 1]].T, tt_edges[:, [0, 1]].T, ht_edges[:, [0, 1]].T, th_edges[:, [0, 1]].T], dim=1), 
        edge_type=torch.cat([hh_edges[:, 2], tt_edges[:, 2], ht_edges[:, 2], th_edges[:, 2]], dim=0),
        num_nodes=num_rels, 
        num_relations=4
    )

    graph.relation_graph = rel_graph
    return graph


class ReifiedGraph(HeteroData):
    """
    """
    def __init__(self, original_node_types: list[str], original_relation_types: list[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_node_types: list[str] = original_node_types
        self.original_relation_types: list[str] = original_relation_types

    def subgraph(self, subset_dict: Dict[NodeType, Tensor], keep_nodes: bool = False) -> 'HeteroData':
        r"""Version of HeteroData.subgraph() which allows to keep nodes in the generated subgraph (it only removes edges)
        """
        data = copy.copy(self)
        subset_dict = copy.copy(subset_dict)

        if not keep_nodes:
            for node_type, subset in subset_dict.items():
                for key, value in self[node_type].items():
                    if key == 'num_nodes':
                        if subset.dtype == torch.bool:
                            data[node_type].num_nodes = int(subset.sum())
                        else:
                            data[node_type].num_nodes = subset.size(0)
                    elif self[node_type].is_node_attr(key):
                        data[node_type][key] = value[subset]
                    else:
                        data[node_type][key] = value

        for edge_type in self.edge_types:
            src, _, dst = edge_type

            src_subset = subset_dict.get(src)
            if src_subset is None:
                src_subset = torch.arange(data[src].num_nodes)
            dst_subset = subset_dict.get(dst)
            if dst_subset is None:
                dst_subset = torch.arange(data[dst].num_nodes)

            edge_index, _, edge_mask = bipartite_subgraph(
                (src_subset, dst_subset),
                self[edge_type].edge_index,
                relabel_nodes=not keep_nodes,
                size=(self[src].num_nodes, self[dst].num_nodes),
                return_edge_mask=True,
            )

            for key, value in self[edge_type].items():
                if key == 'edge_index':
                    data[edge_type].edge_index = edge_index
                elif self[edge_type].is_edge_attr(key):
                    data[edge_type][key] = value[edge_mask]
                else:
                    data[edge_type][key] = value

        return data
    
    def get_relation_instance_repr(self, relation_instance_id: int) -> str:
        """
        Returns a string representation of a relation instance.
        Utility function. Can be used for debugging.
        """
        head_id = self["has_head"].edge_index[:, self["has_head"].edge_index[0] == relation_instance_id][1, 0].item()
        tail_id = self["has_tail"].edge_index[:, self["has_tail"].edge_index[0] == relation_instance_id][1, 0].item()
        relation_type_id = self[("relation_instance", "has_type", "relation_type")].edge_index[:, self[("relation_instance", "has_type", "relation_type")].edge_index[0] == relation_instance_id][1, 0].item()
        head_type_id = self[("node_instance", "has_type", "node_type")].edge_index[:, self[("node_instance", "has_type", "node_type")].edge_index[0] == head_id][1, 0].item()
        tail_type_id = self[("node_instance", "has_type", "node_type")].edge_index[:, self[("node_instance", "has_type", "node_type")].edge_index[0] == tail_id][1, 0].item()
        return f"({head_id}: {self.original_node_types[head_type_id]}) -[{relation_instance_id}:  {self.original_relation_types[relation_type_id][1]}]-> ({tail_id}: {self.original_node_types[tail_type_id]})"


@functional_transform('to_reified_graph')
class ToReifiedGraph(BaseTransform):
    """
    """
    def __init__(self):
        pass

    def forward(
        self,
        data: HeteroData,
        to_homogeneous_kwargs: dict[str, Any] | None = None,
    ) -> ReifiedGraph:
        """
        Convert a heterogeneous graph to a reified graph.
        TODO: describe transformation process
        Warning:
         - the combination of features accross different node/relation types is handled by to_homogeneous(), and thus having a same feature name for a node and a relation is not supported
         - input graph must not contain reverse edges (e.g. head to tail and tail to head)

        Args:
            data: The heterogeneous graph to convert.
            to_homogeneous_kwargs: Keyword arguments to pass to to_homogeneous().
        """
        # we can use node_types because it returns types in the same order as num_nodes_dict (used in to_homogeneous() below)
        node_types = list(data.node_types)
        # we use collect() to make sure we have the same order as the one used in to_homogeneous() below
        relation_types = list(data.collect('edge_index', True).keys())

        if not node_types or not relation_types:
            raise ValueError("The input graph must have at least one node type and one relation type.")

        # we use to_homogeneous() to merge attributes of nodes of different types
        to_homogeneous_kwargs = to_homogeneous_kwargs or {}
        homo_data = data.to_homogeneous(**to_homogeneous_kwargs)
        device = homo_data.edge_index.device
        
        reified_graph = ReifiedGraph(
            original_node_types=node_types,
            original_relation_types=relation_types,
        )

        # create nodes in the reified graph for each node of the original graph
        # node_instance nodes
        reified_graph["node_instance"].num_nodes = homo_data.num_nodes
        for attr_name in homo_data.node_attrs():
            if attr_name == 'node_type':
                continue
            reified_graph["node_instance"][attr_name] = homo_data[attr_name]
        # node_type nodes
        reified_graph["node_type"].num_nodes = len(node_types)
        # node_instance <-> node_type edges
        node_instance_id = list(range(reified_graph["node_instance"].num_nodes))
        node_type_id = [node_types.index(node_type) for node_type in node_types for _ in range(data[node_type].num_nodes)]
        reified_graph["node_instance", "has_type", "node_type"].edge_index = torch.tensor([node_instance_id, node_type_id], dtype=torch.long, device=device)
        reified_graph["node_type", "has_instance", "node_instance"].edge_index = torch.tensor([node_type_id, node_instance_id], dtype=torch.long, device=device)

        # create nodes in the reified graph for each relation of the original graph
        # relation_instance nodes
        reified_graph["relation_instance"].num_nodes = homo_data.num_edges
        for attr_name in homo_data.edge_attrs():
            if attr_name in ['edge_type', 'edge_index']:
                continue
            reified_graph["relation_instance"][attr_name] = homo_data[attr_name]
        # relation_type nodes
        reified_graph["relation_type"].num_nodes = len(relation_types)
        # relation_instance <-> relation_type edges
        relation_instance_id = list(range(reified_graph["relation_instance"].num_nodes))
        relation_type_id = [relation_types.index(relation_type) for relation_type, num_edges in data.num_edges_dict.items() for _ in range(num_edges)]
        reified_graph["relation_instance", "has_type", "relation_type"].edge_index = torch.tensor([relation_instance_id, relation_type_id], dtype=torch.long, device=device)
        reified_graph["relation_type", "has_instance", "relation_instance"].edge_index = torch.tensor([relation_type_id, relation_instance_id], dtype=torch.long, device=device)

        # add head/tail relation in the reified graph to reproduce relations from the original graph
        node_type_offset = {}
        current_offset = 0
        for node_type, num_nodes in data.num_nodes_dict.items():
            node_type_offset[node_type] = current_offset
            current_offset += num_nodes
        head_id, tail_id = [], []
        for relation_type in relation_types:
            head_type = relation_type[0]
            tail_type = relation_type[2]
            head_id.extend((node_type_offset[head_type] + data[relation_type].edge_index[0]).tolist())
            tail_id.extend((node_type_offset[tail_type] + data[relation_type].edge_index[1]).tolist())
        reified_graph["relation_instance", "has_head", "node_instance"].edge_index = torch.tensor([relation_instance_id, head_id], dtype=torch.long, device=device)
        reified_graph["node_instance", "is_head_of", "relation_instance"].edge_index = torch.tensor([head_id, relation_instance_id], dtype=torch.long, device=device)
        reified_graph["relation_instance", "has_tail", "node_instance"].edge_index = torch.tensor([relation_instance_id, tail_id], dtype=torch.long, device=device)
        reified_graph["node_instance", "is_tail_of", "relation_instance"].edge_index = torch.tensor([tail_id, relation_instance_id], dtype=torch.long, device=device)
        
        # add attributes to relation_instance nodes in order to more efficiently remove them during training
        reified_graph["relation_instance"].type_id = torch.tensor(relation_type_id, dtype=torch.long, device=device)
        reified_graph["relation_instance"].head_id = torch.tensor(head_id, dtype=torch.long, device=device)
        reified_graph["relation_instance"].tail_id = torch.tensor(tail_id, dtype=torch.long, device=device)

        return reified_graph
