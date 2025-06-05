from typing import Callable, Dict, List, Optional, Union, Any, Iterator
from collections.abc import Sequence

from torch_geometric.data import Data, HeteroData
from torch_geometric.sampler import NeighborSampler
from torch_geometric.sampler.base import SubgraphType
from torch_geometric.typing import EdgeType, OptTensor

import torch
from torch import Tensor

from torch_geometric.loader.base import DataLoaderIterator
from torch_geometric.loader.mixin import AffinityMixin
from torch_geometric.loader.utils import (
    filter_custom_store,
    filter_data,
    filter_hetero_data,
    infer_filter_per_worker,
)
from torch_geometric.sampler import (
    HeteroSamplerOutput,
    NodeSamplerInput,
    SamplerOutput,
)

from ultra.tasks import ReifiedGraph


class NeighborSamplerMultipleSeedTypes(NeighborSampler):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def sample_from_nodes_multiple_seed_types(
        self,
        inputs: list[NodeSamplerInput],
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        seed = {inputs_item.input_type: inputs_item.node for inputs_item in inputs}
        seed_time = None
        out = self._sample(seed, seed_time)
        # out.metadata = {inputs_item.input_type: (inputs_item.input_id, inputs_item.time) for inputs_item in inputs}
        out.metadata = (inputs[0].input_id, inputs[0].time)
        return out


# adapted from torch_geometric.loader.utils.get_input_nodes
def get_input_relation_instance_nodes(
    data: HeteroData,
    input_relations: OptTensor,
) -> Sequence:
    def to_index(tensor):
        if isinstance(tensor, Tensor) and tensor.dtype == torch.bool:
            return tensor.nonzero(as_tuple=False).view(-1)
        if not isinstance(tensor, Tensor):
            return torch.tensor(tensor, dtype=torch.long)
        return tensor

    assert isinstance(data, HeteroData)

    if input_relations is None:
        return torch.arange(data["relation_instance"].num_nodes)
    return to_index(input_relations)



# adapted from NeighborLoader and its superclass NodeLoader
# TODO: when it works, make it a subclass of NeighborLoader and remove the
#       duplicated code
class ReifiedLinkNeighborLoader(torch.utils.data.DataLoader, AffinityMixin):
    def __init__(
        self,
        data: ReifiedGraph,
        num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
        # input_nodes: InputNodes = None,
        fact_relations: OptTensor = None,
        target_relations: OptTensor = None,
        # input_time: OptTensor = None,
        replace: bool = False,
        subgraph_type: Union[SubgraphType, str] = 'directional',
        disjoint: bool = False,
        temporal_strategy: str = 'uniform',
        time_attr: Optional[str] = None,
        weight_attr: Optional[str] = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        is_sorted: bool = False,
        filter_per_worker: Optional[bool] = None,
        neighbor_sampler: Optional[NeighborSamplerMultipleSeedTypes] = None,
        directed: bool = True,  # Deprecated.
        num_negative: int = 32,
        **kwargs,
    ):

        # convert fact_relations and target_relations to lists of indices
        fact_relations = get_input_relation_instance_nodes(data, fact_relations)
        target_relations = get_input_relation_instance_nodes(data, target_relations)

        # build a subgraph from data with only fact relations
        self.data = data.subgraph({"relation_instance": fact_relations}, keep_nodes=True)

        # store ids of nodes connected to the target relation_instance nodes
        self.target_relation_instance_id = target_relations
        self.target_relation_type_id = data["relation_instance"].type_id[target_relations]
        self.target_relation_head_id = data["relation_instance"].head_id[target_relations]
        self.target_relation_tail_id = data["relation_instance"].tail_id[target_relations]
        
        # from NeighborLoader __init__()

        if neighbor_sampler is None:
            neighbor_sampler = NeighborSamplerMultipleSeedTypes(
                self.data,
                num_neighbors=num_neighbors,
                replace=replace,
                subgraph_type=subgraph_type,
                disjoint=disjoint,
                temporal_strategy=temporal_strategy,
                time_attr=time_attr,
                weight_attr=weight_attr,
                is_sorted=is_sorted,
                share_memory=kwargs.get('num_workers', 0) > 0,
                directed=directed,
            )
        
        # from NodeLoader __init__()
        
        if filter_per_worker is None:
            filter_per_worker = infer_filter_per_worker(data)

        # Remove for PyTorch Lightning:
        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)

        self.node_sampler = neighbor_sampler  # node_sampler
        self.transform = transform
        self.transform_sampler_output = transform_sampler_output
        self.filter_per_worker = filter_per_worker
        self.custom_cls = None  # custom_cls        
        self.num_negative = num_negative

        iterator = range(target_relations.size(0))
        super().__init__(iterator, collate_fn=self.collate_fn, **kwargs)

    def __call__(
        self,
        index: Union[Tensor, List[int]],
    ) -> Union[Data, HeteroData]:
        r"""Samples a subgraph from a batch of input nodes."""
        out = self.collate_fn(index)
        if not self.filter_per_worker:
            out = self.filter_fn(out)
        return out

    def collate_fn(self, index: Union[Tensor, List[int]]) -> Any:
        r"""Samples a subgraph from a batch of input nodes."""
        if isinstance(index, (list, tuple)):
            index = torch.tensor(index)
        target_relation_type_id = self.target_relation_type_id[index]
        target_relation_head_id = self.target_relation_head_id[index]
        target_relation_tail_id = self.target_relation_tail_id[index]
        
        # add negative samples
        if self.num_negative > 0:
            # TODO: support strict sampling of reified graphs

            target_relation_type_id = target_relation_type_id.repeat_interleave(1 + self.num_negative)
            num_negative_tails = self.num_negative // 2
            num_negative_heads = self.num_negative - num_negative_tails
            target_relation_head_id = torch.cat([
                target_relation_head_id,
                target_relation_head_id.repeat_interleave(num_negative_tails),
                torch.randint(
                    self.data["node_instance"].num_nodes, (len(index) * num_negative_heads,)
                ),
            ])
            target_relation_tail_id = torch.cat([
                target_relation_tail_id,
                torch.randint(
                    self.data["node_instance"].num_nodes, (len(index) * num_negative_heads,)
                ),
                target_relation_tail_id.repeat_interleave(num_negative_tails),
            ])

        # node_instance_ids = torch.cat([head_ids, tail_ids])
        node_instance_input = NodeSamplerInput(
            input_id=index,
            node=torch.cat([target_relation_head_id, target_relation_tail_id]),
            input_type="node_instance",
        )
        relation_type_input = NodeSamplerInput(
            input_id=index,
            node=target_relation_type_id,
            input_type="relation_type",
        )

        out = self.node_sampler.sample_from_nodes_multiple_seed_types([relation_type_input, node_instance_input])

        if self.filter_per_worker:  # Execute `filter_fn` in the worker process
            out = self.filter_fn(out)

        # TODO: check what happens with below fields when filter_fn is called in the main process
        out.type_ids = target_relation_type_id
        out.head_ids = target_relation_head_id
        out.tail_ids = target_relation_tail_id

        # sampling is performed on the fact relations only, but in some cases (typically during training), 
        # fact relations are also used as target relations
        # For this reason, we need to remove from the sampled graph the target relation_instance nodes which
        # are used as seeds for this batch
        # TODO: find out if we can replace this step with more efficient ones, like:
        #  - deleting edges by directly manipulating edge_index
        #  - setting num_neighbors so that the target relation_instance nodes are not sampled (this can work only for some edges and can be complementary to the above)
        relation_instance_to_keep = torch.arange(out["relation_instance"].num_nodes, dtype=torch.long)[torch.stack([out["relation_instance"].n_id!=i for i in self.target_relation_instance_id[index]]).all(dim=0)]
        out = out.subgraph({"relation_instance": relation_instance_to_keep}, keep_nodes=True)

        # FIXME: something is odd: when comparing out before and after call to subgraph(), 3 has_head relations have disapeared instead of two (batch sier is 2)
        # this is because relation_instance with id 0 was related to 2 node_instance nodes instead of 1
        # this should be investigated        

        return out

    def filter_fn(
        self,
        out: Union[SamplerOutput, HeteroSamplerOutput],
    ) -> Union[Data, HeteroData]:
        r"""Joins the sampled nodes with their corresponding features,
        returning the resulting :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object to be used downstream.
        """
        if self.transform_sampler_output:
            out = self.transform_sampler_output(out)

        if isinstance(out, SamplerOutput):
            data = filter_data(self.data, out.node, out.row, out.col, out.edge,
                               self.node_sampler.edge_permutation)

            if 'n_id' not in data:
                data.n_id = out.node
            if out.edge is not None and 'e_id' not in data:
                edge = out.edge.to(torch.long)
                perm = self.node_sampler.edge_permutation
                data.e_id = perm[edge] if perm is not None else edge

            data.batch = out.batch
            data.num_sampled_nodes = out.num_sampled_nodes
            data.num_sampled_edges = out.num_sampled_edges

            data.input_id = out.metadata[0]
            data.seed_time = out.metadata[1]
            data.batch_size = out.metadata[0].size(0)

        elif isinstance(out, HeteroSamplerOutput):
            if isinstance(self.data, HeteroData):
                data = filter_hetero_data(self.data, out.node, out.row,
                                          out.col, out.edge,
                                          self.node_sampler.edge_permutation)
            else:  # Tuple[FeatureStore, GraphStore]
                data = filter_custom_store(*self.data, out.node, out.row,
                                           out.col, out.edge, self.custom_cls)

            for key, node in out.node.items():
                if 'n_id' not in data[key]:
                    data[key].n_id = node

            for key, edge in (out.edge or {}).items():
                if edge is not None and 'e_id' not in data[key]:
                    edge = edge.to(torch.long)
                    perm = self.node_sampler.edge_permutation[key]
                    data[key].e_id = perm[edge] if perm is not None else edge

            data.set_value_dict('batch', out.batch)
            data.set_value_dict('num_sampled_nodes', out.num_sampled_nodes)
            data.set_value_dict('num_sampled_edges', out.num_sampled_edges)

            data.input_id = out.metadata[0]
            data.seed_time = out.metadata[1]
            data.batch_size = out.metadata[0].size(0)

        else:
            raise TypeError(f"'{self.__class__.__name__}'' found invalid "
                            f"type: '{type(out)}'")

        return data if self.transform is None else self.transform(data)

    def _get_iterator(self) -> Iterator:
        if self.filter_per_worker:
            return super()._get_iterator()

        # if not self.is_cuda_available and not self.cpu_affinity_enabled:
        # TODO: Add manual page for best CPU practices
        # link = ...
        # Warning('Dataloader CPU affinity opt is not enabled, consider '
        #          'switching it on with enable_cpu_affinity() or see CPU '
        #          f'best practices for PyG [{link}])')

        # Execute `filter_fn` in the main process:
        return DataLoaderIterator(super()._get_iterator(), self.filter_fn)

    def __enter__(self):
        return self

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
