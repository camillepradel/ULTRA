import os
import sys
import math
import pprint
from itertools import islice
from typing import Literal

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from script.reified_link_neighbor_loader import ReifiedLinkNeighborLoader, get_input_relation_instance_nodes
from ultra.tasks import ReifiedGraph
from ultra import tasks, util
from ultra.models import ReiFM


separator = ">" * 30
line = "-" * 30


def train_and_validate(cfg, model, data: ReifiedGraph, device, logger, batch_per_epoch=None):
    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()
    
    train_loader = ReifiedLinkNeighborLoader(
        data,
        # num_neighbors={
        #     ("node_instance", "has_type", "node_type"): [1, 1, 1],
        #     ("node_type", "has_instance", "node_instance"): [30, 30, 30],
        #     ("relation_instance", "has_type", "relation_type"): [1, 1, 1],
        #     ("relation_type", "has_instance", "relation_instance"): [30, 30, 30],
        #     ("relation_instance", "has_head", "node_instance"): [1, 1, 1],
        #     ("node_instance", "is_head_of", "relation_instance"): [10, 10, 10],
        #     ("relation_instance", "has_tail", "node_instance"): [1, 1, 1],
        #     ("node_instance", "is_tail_of", "relation_instance"): [10, 10, 10],
        # },
        # num_neighbors={
        #     ("node_instance", "has_type", "node_type"): [-1] * cfg.model.hop_count,
        #     ("node_type", "has_instance", "node_instance"): [-1] * cfg.model.hop_count,
        #     ("relation_instance", "has_type", "relation_type"): [-1] * cfg.model.hop_count,
        #     ("relation_type", "has_instance", "relation_instance"): [-1] * cfg.model.hop_count,
        #     ("relation_instance", "has_head", "node_instance"): [-1] * cfg.model.hop_count,
        #     ("node_instance", "is_head_of", "relation_instance"): [-1] * cfg.model.hop_count,
        #     ("relation_instance", "has_tail", "node_instance"): [-1] * cfg.model.hop_count,
        #     ("node_instance", "is_tail_of", "relation_instance"): [-1] * cfg.model.hop_count,
        # },
        num_neighbors={
            ("node_instance", "has_type", "node_type"): [30] * cfg.model.hop_count,
            ("node_type", "has_instance", "node_instance"): [5] * cfg.model.hop_count,
            ("relation_instance", "has_type", "relation_type"): [30] * cfg.model.hop_count,
            ("relation_type", "has_instance", "relation_instance"): [5] * cfg.model.hop_count,
            ("relation_instance", "has_head", "node_instance"): [10] * cfg.model.hop_count,
            ("node_instance", "is_head_of", "relation_instance"): [5] * cfg.model.hop_count,
            ("relation_instance", "has_tail", "node_instance"): [10] * cfg.model.hop_count,
            ("node_instance", "is_tail_of", "relation_instance"): [5] * cfg.model.hop_count,
        },
        fact_relations=data["relation_instance"].train_fact_mask,
        target_relations=data["relation_instance"].train_target_mask,
        batch_size=cfg.train.batch_size,
        num_negative=cfg.task.num_negative,
    )

    batch_per_epoch = batch_per_epoch or len(train_loader)

    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)
    num_params = sum(p.numel() for p in model.parameters())
    logger.warning(line)
    logger.warning(f"Number of parameters: {num_params}")

    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    step = math.ceil(cfg.train.num_epoch / 10)
    best_result = float("-inf")
    best_epoch = -1

    batch_id = 0
    for i in range(0, cfg.train.num_epoch, step):
        parallel_model.train()
        for epoch in range(i, min(cfg.train.num_epoch, i + step)):
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Epoch %d begin" % epoch)

            losses = []
            for batch in islice(train_loader, batch_per_epoch):
                
                
                
                # TODO: delete the relation_instance node of the relation to predict when training
                pred = parallel_model(batch)
                target = torch.cat((
                    torch.ones(batch.batch_size, dtype=pred.dtype, device=pred.device),
                    torch.zeros(batch.batch_size * cfg.task.num_negative, dtype=pred.dtype, device=pred.device),
                ))
                
                loss = F.binary_cross_entropy(pred, target, reduction="none").mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if util.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                    logger.warning(separator)
                    logger.warning("binary cross entropy: %g" % loss)
                losses.append(loss.item())
                batch_id += 1

            if util.get_rank() == 0:
                avg_loss = sum(losses) / len(losses)
                logger.warning(separator)
                logger.warning("Epoch %d end" % epoch)
                logger.warning(line)
                logger.warning("average binary cross entropy: %g" % avg_loss)

        epoch = min(cfg.train.num_epoch, i + step)
        if rank == 0:
            logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, "model_epoch_%d.pth" % epoch)
        util.synchronize()

        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        result = test(cfg, model, data, split='valid', device=device, logger=logger)
        if result > best_result:
            best_result = result
            best_epoch = epoch

    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device)
    model.load_state_dict(state["model"])
    util.synchronize()


@torch.no_grad()
def test(cfg, model, data: ReifiedGraph, split: Literal['valid', 'test'], device, logger, return_metrics=False):
    world_size = util.get_world_size()
    rank = util.get_rank()

    if split == 'valid':
        fact_relations = data["relation_instance"].valid_fact_mask
        target_relations = data["relation_instance"].valid_target_mask
    elif split == 'test':
        fact_relations = data["relation_instance"].test_fact_mask
        target_relations = data["relation_instance"].test_target_mask
        
    # TODO: use loader

    # convert fact_relations and target_relations to lists of indices
    fact_relations = get_input_relation_instance_nodes(data, fact_relations)

    test_triplets = torch.stack([
        data["relation_instance"].head_id[target_relations],
        data["relation_instance"].tail_id[target_relations],
        data["relation_instance"].type_id[target_relations]
    ], dim=1)
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank, shuffle=False)
    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

    # build a subgraph from data with only fact relations
    data = data.subgraph({"relation_instance": fact_relations}, keep_nodes=True)



    
    model.eval()
    rankings = []
    num_negatives = []
    tail_rankings, num_tail_negs = [], []  # for explicit tail-only evaluation needed for 5 datasets
    
    x_dict = model(data)
    
    for batch in test_loader:
        
        batch_size = batch.size(0)
        t_batch, h_batch = tasks.all_negative(data, batch)
        
        # tail prediction
        heads = t_batch[:, :, 0].reshape(-1)
        tails = t_batch[:, :, 1].reshape(-1)
        rels = t_batch[:, :, 2].reshape(-1)
        
        heads_repr = x_dict["node_instance"].index_select(0, heads)
        tails_repr = x_dict["node_instance"].index_select(0, tails)
        rels_repr = x_dict["relation_instance"].index_select(0, rels)
        
        t_pred = model.mlp(
            torch.cat((heads_repr, tails_repr, rels_repr), dim=1)
        )
        t_pred = t_pred.reshape(batch_size, -1)
        
        # head prediction
        heads = h_batch[:, :, 0].reshape(-1)
        tails = h_batch[:, :, 1].reshape(-1)
        rels = h_batch[:, :, 2].reshape(-1)
        
        heads_repr = x_dict["node_instance"].index_select(0, heads)
        tails_repr = x_dict["node_instance"].index_select(0, tails)
        rels_repr = x_dict["relation_instance"].index_select(0, rels)
        
        h_pred = model.mlp(
            torch.cat((heads_repr, tails_repr, rels_repr), dim=1)
        )
        h_pred = h_pred.reshape(batch_size, -1)

        if True:  # filtered_data is None:
            # TODO
            # t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
            t_mask, h_mask = torch.ones_like(t_pred, dtype=torch.bool), torch.ones_like(h_pred, dtype=torch.bool)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)

        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

        tail_rankings += [t_ranking]
        num_tail_negs += [num_t_negative]

    ranking = torch.cat(rankings)
    num_negative = torch.cat(num_negatives)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)

    # ugly repetitive code for tail-only ranks processing
    tail_ranking = torch.cat(tail_rankings)
    num_tail_neg = torch.cat(num_tail_negs)
    all_size_t = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_t[rank] = len(tail_ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_size_t, op=dist.ReduceOp.SUM)

    # obtaining all ranks 
    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative

    # the same for tails-only ranks
    cum_size_t = all_size_t.cumsum(0)
    all_ranking_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
    all_ranking_t[cum_size_t[rank] - all_size_t[rank]: cum_size_t[rank]] = tail_ranking
    all_num_negative_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
    all_num_negative_t[cum_size_t[rank] - all_size_t[rank]: cum_size_t[rank]] = num_tail_neg
    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_ranking_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative_t, op=dist.ReduceOp.SUM)

    metrics = {}
    if rank == 0:
        for metric in cfg.task.metric:
            if "-tail" in metric:
                _metric_name, direction = metric.split("-")
                if direction != "tail":
                    raise ValueError("Only tail metric is supported in this mode")
                _ranking = all_ranking_t
                _num_neg = all_num_negative_t
            else:
                _ranking = all_ranking 
                _num_neg = all_num_negative 
                _metric_name = metric
            
            if _metric_name == "mr":
                score = _ranking.float().mean()
            elif _metric_name == "mrr":
                score = (1 / _ranking.float()).mean()
            elif _metric_name.startswith("hits@"):
                values = _metric_name[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    # unbiased estimation
                    fp_rate = (_ranking - 1).float() / _num_neg
                    score = 0
                    for i in range(threshold):
                        # choose i false positive from num_sample - 1 negatives
                        num_comb = math.factorial(num_sample - 1) / \
                                   math.factorial(i) / math.factorial(num_sample - i - 1)
                        score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                    score = score.mean()
                else:
                    score = (_ranking <= threshold).float().mean()
            logger.warning("%s: %g" % (metric, score))
            metrics[metric] = score
    mrr = (1 / all_ranking.float()).mean()

    return mrr if not return_metrics else metrics


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
    
    task_name = cfg.task["name"]
    dataset = util.build_dataset(cfg)
    device = util.get_device(cfg)

    data = dataset[0]
    data = data.to(device)

    model = ReiFM(
        data=data,
        hidden_channels=64,
        hop_count=cfg.model.hop_count,
    )

    if "checkpoint" in cfg and cfg.checkpoint is not None:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])

    #model = pyg.compile(model, dynamic=True)
    model = model.to(device)
    
    # TODO: make sure below behavior is implemented in all dataset constructors and remove it from here
    if task_name == "InductiveInference":
        # filtering for inductive datasets
        # Grail, MTDEA, HM datasets have validation sets based off the training graph
        # ILPC, Ingram have validation sets from the inference graph
        # filtering dataset should contain all true edges (base graph + (valid) + test) 
        if "ILPC" in cfg.dataset['class'] or "Ingram" in cfg.dataset['class']:
            # add inference, valid, test as the validation and test filtering graphs
            full_inference_edges = torch.cat([valid_data.edge_index, valid_data.target_edge_index, test_data.target_edge_index], dim=1)
            full_inference_etypes = torch.cat([valid_data.edge_type, valid_data.target_edge_type, test_data.target_edge_type])
            test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)
            val_filtered_data = test_filtered_data
        else:
            # # test filtering graph: inference edges + test edges
            # test_filtered_data = Data(
            #     edge_index=torch.cat([test_data.edge_index, test_data.target_edge_index], dim=1),
            #     edge_type=torch.cat([test_data.edge_type, test_data.target_edge_type]),
            #     num_nodes=test_data.num_nodes
            # )

            # # validation filtering graph: train edges + validation edges
            # val_filtered_data = Data(
            #     edge_index=torch.cat([train_data.edge_index, valid_data.target_edge_index], dim=1),
            #     edge_type=torch.cat([train_data.edge_type, valid_data.target_edge_type])
            # )
            
            val_filtered_data = test_filtered_data = None # TODO
    else:
        # for transductive setting, use the whole graph for filtered ranking
        filtered_data = Data(edge_index=dataset._data.target_edge_index, edge_type=dataset._data.target_edge_type, num_nodes=dataset[0].num_nodes)
        val_filtered_data = test_filtered_data = filtered_data
    
    # val_filtered_data = val_filtered_data.to(device)
    # test_filtered_data = test_filtered_data.to(device)
    
    train_and_validate(cfg, model, data, device=device, batch_per_epoch=cfg.train.batch_per_epoch, logger=logger)
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on valid")
    test(cfg, model, data, split='valid', device=device, logger=logger)
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")
    test(cfg, model, data, split='test', device=device, logger=logger)
