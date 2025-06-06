from functools import reduce
from torch_scatter import scatter_add
from torch_geometric.data import Data, HeteroData
import torch


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


def build_reified_graph(graph):

    original_edge_index, original_edge_type = graph.edge_index, graph.edge_type
    original_num_node_instances = graph.num_nodes
    original_num_rel_types = graph.num_relations
    original_num_rel_instances = original_edge_index.shape[1]
    original_heads = original_edge_index[0]
    original_tails = original_edge_index[1]
    device = original_edge_index.device
    
    # node types:
    # NODE_INSTANCE = 0
    # NODE_TYPE = 1
    # RELATION_INSTANCE = 2
    # RELATION_TYPE = 3
    NODE_INSTANCE = "node_instance"
    NODE_TYPE = "node_type"
    RELATION_INSTANCE = "relation_instance"
    RELATION_TYPE = "relation_type"

    # relation types:
    # HAS_TYPE = 0
    # HAS_INSTANCE = 1
    # HAS_HEAD = 2
    # IS_HEAD_OF = 3
    # HAS_TAIL = 4
    # IS_TAIL_OF = 5
    HAS_TYPE = "has_type"
    HAS_INSTANCE = "has_instance"
    HAS_HEAD = "has_head"
    IS_HEAD_OF = "is_head_of"
    HAS_TAIL = "has_tail"
    IS_TAIL_OF = "is_tail_of"

    # reified_edge_index = []
    # reified_edge_type = []
    # reified_node_type = []
    
    # reified node nodes
    
    # nodes from the original graph are now node instances
    # reified_node_type.extend([NODE_INSTANCE] * original_num_node_instances)
    
    # create a node_type node for each node type
    # TODO: support as input heterogeneous graphs where there is multiple node types
    # reified_node_type_index = len(reified_node_type)
    # reified_node_type.append(NODE_TYPE)

    # create relations to state the type of each node_instance node
    # reified_edge_index.extend([(node_instance_index, reified_node_type_index) for node_instance_index in range(original_num_node_instances)])
    # reified_edge_index.extend([(reified_node_type_index, node_instance_index) for node_instance_index in range(original_num_node_instances)])
    # reified_edge_type.extend([HAS_TYPE] * original_num_node_instances + [HAS_INSTANCE] * original_num_node_instances)

    # reified relation nodes

    # create a relation_instance node for each relation
    # reified_first_relation_instance_index = len(reified_node_type)
    # reified_node_type.extend([RELATION_INSTANCE] * original_num_rel_instances)

    # create a relation_type node for each relation type
    # reified_first_relation_type_index = len(reified_node_type)
    # reified_node_type.extend([RELATION_TYPE] * original_num_rel_types)

    # create relations to state the type of each relation_type node
    # reified_edge_index.extend([(reified_first_relation_instance_index + relation_instance_index, reified_first_relation_type_index + original_edge_type[relation_instance_index]) for relation_instance_index in range(original_num_rel_instances)])
    # reified_edge_index.extend([(reified_first_relation_type_index + original_edge_type[relation_instance_index], reified_first_relation_instance_index + relation_instance_index) for relation_instance_index in range(original_num_rel_instances)])
    # reified_edge_type.extend([HAS_TYPE] * original_num_rel_instances + [HAS_INSTANCE] * original_num_rel_instances)

    # reification relations
    # reified_edge_index.extend([(reified_first_relation_instance_index + relation_instance_index, original_heads[relation_instance_index]) for relation_instance_index in range(original_num_rel_instances)])
    # reified_edge_index.extend([(original_heads[relation_instance_index], reified_first_relation_instance_index + relation_instance_index) for relation_instance_index in range(original_num_rel_instances)])
    # reified_edge_type.extend([HAS_HEAD] * original_num_rel_instances + [IS_HEAD_OF] * original_num_rel_instances)
    # reified_edge_index.extend([(reified_first_relation_instance_index + relation_instance_index, original_tails[relation_instance_index]) for relation_instance_index in range(original_num_rel_instances)])
    # reified_edge_index.extend([(original_tails[relation_instance_index], reified_first_relation_instance_index + relation_instance_index) for relation_instance_index in range(original_num_rel_instances)])
    # reified_edge_type.extend([HAS_TAIL] * original_num_rel_instances + [IS_TAIL_OF] * original_num_rel_instances)
    
    # reified graph
    # reified_edge_index = torch.tensor(reified_edge_index, dtype=torch.long, device=device).T
    # reified_edge_type = torch.tensor(reified_edge_type, dtype=torch.long, device=device)
    # reified_node_type = torch.tensor(reified_node_type, dtype=torch.long, device=device)
    # num_reified_nodes = len(reified_node_type)
    # num_reified_edges = len(reified_edge_index[0])
    
    reified_graph = HeteroData(
        target_edge_index=graph.target_edge_index,
        target_edge_type=graph.target_edge_type,
    )

    # TODO: make sure saving node indices (each line below with ...node_id = torch.arange(...)) is useful

    # create nodes in the reified graph for each node of the original graph
    reified_graph[NODE_INSTANCE].node_id = torch.arange(original_num_node_instances)
    reified_graph[NODE_TYPE].node_id = torch.arange(1) # TODO: support as input heterogeneous graphs where there is multiple node types
    reified_graph[NODE_INSTANCE, HAS_TYPE, NODE_TYPE].edge_index = torch.tensor([(node_instance_index, 0) for node_instance_index in range(original_num_node_instances)], dtype=torch.long, device=device).T
    reified_graph[NODE_TYPE, HAS_INSTANCE, NODE_INSTANCE].edge_index = torch.tensor([(0, node_instance_index) for node_instance_index in range(original_num_node_instances)], dtype=torch.long, device=device).T

    # create nodes in the reified graph for each relation of the original graph
    reified_graph[RELATION_INSTANCE].node_id = torch.arange(original_num_rel_instances)
    reified_graph[RELATION_TYPE].node_id = torch.arange(original_num_rel_types)
    reified_graph[RELATION_INSTANCE, HAS_TYPE, RELATION_TYPE].edge_index = torch.tensor([(relation_instance_index, original_edge_type[relation_instance_index]) for relation_instance_index in range(original_num_rel_instances)], dtype=torch.long, device=device).T
    reified_graph[RELATION_TYPE, HAS_INSTANCE, RELATION_INSTANCE].edge_index = torch.tensor([(original_edge_type[relation_instance_index], relation_instance_index) for relation_instance_index in range(original_num_rel_instances)], dtype=torch.long, device=device).T

    # add head/tail relation in the reified graph to reproduce relations from the original graph
    reified_graph[RELATION_INSTANCE, HAS_HEAD, NODE_INSTANCE].edge_index = torch.tensor([(relation_instance_index, original_heads[relation_instance_index]) for relation_instance_index in range(original_num_rel_instances)], dtype=torch.long, device=device).T
    reified_graph[NODE_INSTANCE, IS_HEAD_OF, RELATION_INSTANCE].edge_index = torch.tensor([(original_heads[relation_instance_index], relation_instance_index) for relation_instance_index in range(original_num_rel_instances)], dtype=torch.long, device=device).T
    reified_graph[RELATION_INSTANCE, HAS_TAIL, NODE_INSTANCE].edge_index = torch.tensor([(relation_instance_index, original_tails[relation_instance_index]) for relation_instance_index in range(original_num_rel_instances)], dtype=torch.long, device=device).T
    reified_graph[NODE_INSTANCE, IS_TAIL_OF, RELATION_INSTANCE].edge_index = torch.tensor([(original_tails[relation_instance_index], relation_instance_index) for relation_instance_index in range(original_num_rel_instances)], dtype=torch.long, device=device).T

    return reified_graph


