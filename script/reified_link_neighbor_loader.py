from typing import Callable, Dict, List, Optional, Tuple, Union

from torch_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from torch_geometric.loader.node_loader import NodeLoader
from torch_geometric.sampler import NeighborSampler
from torch_geometric.sampler.base import SubgraphType
from torch_geometric.typing import EdgeType, InputNodes, OptTensor

from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from torch_geometric.loader.base import DataLoaderIterator
from torch_geometric.loader.mixin import AffinityMixin
from torch_geometric.loader.utils import (
    filter_custom_store,
    filter_data,
    filter_hetero_data,
    get_input_nodes,
    infer_filter_per_worker,
)
from torch_geometric.sampler import (
    BaseSampler,
    HeteroSamplerOutput,
    NodeSamplerInput,
    SamplerOutput,
)
from torch_geometric.typing import InputNodes, OptTensor


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


# adapted from NeighborLoader and its superclass NodeLoader
# TODO: when it works, make it a subclass of NeighborLoader and remove the
#       duplicated code
class ReifiedLinkNeighborLoader(torch.utils.data.DataLoader, AffinityMixin):
    def __init__(
        self,
        data: HeteroData,
        num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
        # input_nodes: InputNodes = None, -> replaced by data.target_edge_index and data.target_edge_type
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
        
        self.num_negative = num_negative
        
        # from NeighborLoader __init__()
        
        # if input_time is not None and time_attr is None:
        #     raise ValueError("Received conflicting 'input_time' and "
        #                      "'time_attr' arguments: 'input_time' is set "
        #                      "while 'time_attr' is not set.")

        if neighbor_sampler is None:
            neighbor_sampler = NeighborSamplerMultipleSeedTypes(
                data,
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

        # super().__init__(
        #     data=data,
        #     node_sampler=neighbor_sampler,
        #     input_nodes=input_nodes,
        #     input_time=input_time,
        #     transform=transform,
        #     transform_sampler_output=transform_sampler_output,
        #     filter_per_worker=filter_per_worker,
        #     **kwargs,
        # )
        
        # from NodeLoader __init__()
        
        if filter_per_worker is None:
            filter_per_worker = infer_filter_per_worker(data)

        # Remove for PyTorch Lightning:
        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)

        # # Get node type (or `None` for homogeneous graphs):
        # input_type, input_nodes = get_input_nodes(data, input_nodes)

        self.data = data
        self.node_sampler = neighbor_sampler  # node_sampler
        self.transform = transform
        self.transform_sampler_output = transform_sampler_output
        self.filter_per_worker = filter_per_worker
        self.custom_cls = None  # custom_cls

        self.input_data_node_instance = NodeSamplerInput(
            input_id=None,  # input_id,
            node=data.target_edge_index.T,
            input_type="node_instance",
        )

        self.input_data_relation_type = NodeSamplerInput(
            input_id=None,  # input_id,
            node=data.target_edge_type,
            input_type="relation_type"
        )

        iterator = range(data.target_edge_index.size(1))
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
        input_data_node_instance: NodeSamplerInput = self.input_data_node_instance[index]
        input_data_relation_type: NodeSamplerInput = self.input_data_relation_type[index]
        
        # add negative samples
        if self.num_negative > 0:
            # TODO: support strict sampling to reified graphs
        
            # generate negative samples for node instances
            neg_node_instance = torch.randint(
                self.data["node_instance"].num_nodes, (len(index) * self.num_negative, 2)
            )
            input_data_node_instance.node = torch.cat(
                [input_data_node_instance.node, neg_node_instance]
            )

            # generate negative samples for relation types
            neg_relation_type = torch.randint(
                self.data["relation_type"].num_nodes, (len(index) * self.num_negative,)
            )
            input_data_relation_type.node = torch.cat(
                [input_data_relation_type.node, neg_relation_type]
            )

        # flatten node instance indices
        input_data_node_instance.node = input_data_node_instance.node.T.reshape((-1,))

        out = self.node_sampler.sample_from_nodes_multiple_seed_types([input_data_node_instance, input_data_relation_type])

        if self.filter_per_worker:  # Execute `filter_fn` in the worker process
            out = self.filter_fn(out)

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
