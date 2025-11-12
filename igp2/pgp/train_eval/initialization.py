# Import datasets
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from igp2.pgp.datasets.interface import TrajectoryDataset
from igp2.pgp.datasets.nuScenes.nuScenes_raster import NuScenesRaster
from igp2.pgp.datasets.nuScenes.nuScenes_vector import NuScenesVector
from igp2.pgp.datasets.nuScenes.nuScenes_graphs import NuScenesGraphs

# Import models
from igp2.pgp.models.model import PredictionModel
from igp2.pgp.models.encoders.raster_encoder import RasterEncoder
from igp2.pgp.models.encoders.polyline_subgraph import PolylineSubgraphs
from igp2.pgp.models.encoders.pgp_encoder import PGPEncoder
from igp2.pgp.models.aggregators.concat import Concat
from igp2.pgp.models.aggregators.global_attention import GlobalAttention
from igp2.pgp.models.aggregators.goal_conditioned import GoalConditioned
from igp2.pgp.models.aggregators.pgp import PGP
from igp2.pgp.models.decoders.mtp import MTP
from igp2.pgp.models.decoders.multipath import Multipath
from igp2.pgp.models.decoders.covernet import CoverNet
from igp2.pgp.models.decoders.lvm import LVM

# Import metrics
from igp2.pgp.metrics.mtp_loss import MTPLoss
from igp2.pgp.metrics.min_ade import MinADEK
from igp2.pgp.metrics.min_fde import MinFDEK
from igp2.pgp.metrics.miss_rate import MissRateK
from igp2.pgp.metrics.covernet_loss import CoverNetLoss
from igp2.pgp.metrics.pi_bc import PiBehaviorCloning
from igp2.pgp.metrics.goal_pred_nll import GoalPredictionNLL

from typing import List, Dict, Union


# Datasets
def initialize_dataset(dataset_type: str, args: List) -> TrajectoryDataset:
    """
    Helper function to initialize appropriate dataset by dataset type string
    """
    # TODO: Add more datasets as implemented
    dataset_classes = {'nuScenes_single_agent_raster': NuScenesRaster,
                       'nuScenes_single_agent_vector': NuScenesVector,
                       'nuScenes_single_agent_graphs': NuScenesGraphs,
                       }
    return dataset_classes[dataset_type](*args)


def get_specific_args(dataset_name: str, data_root: str, version: str = None) -> List:
    """
    Helper function to get dataset specific arguments.
    """
    # TODO: Add more datasets as implemented
    specific_args = []
    if dataset_name == 'nuScenes':
        ns = NuScenes(version, dataroot=data_root)
        pred_helper = PredictHelper(ns)
        specific_args.append(pred_helper)

    return specific_args


# Models
def initialize_prediction_model(encoder_type: str, aggregator_type: str, decoder_type: str,
                                encoder_args: Dict, aggregator_args: Union[Dict, None], decoder_args: Dict):
    """
    Helper function to initialize appropriate encoder, aggegator and decoder models
    """
    encoder = initialize_encoder(encoder_type, encoder_args)
    aggregator = initialize_aggregator(aggregator_type, aggregator_args)
    decoder = initialize_decoder(decoder_type, decoder_args)
    model = PredictionModel(encoder, aggregator, decoder)

    return model


def initialize_encoder(encoder_type: str, encoder_args: Dict):
    """
    Initialize appropriate encoder by type.
    """
    # TODO: Update as we add more encoder types
    encoder_mapping = {
        'raster_encoder': RasterEncoder,
        'polyline_subgraphs': PolylineSubgraphs,
        'pgp_encoder': PGPEncoder
    }

    return encoder_mapping[encoder_type](encoder_args)


def initialize_aggregator(aggregator_type: str, aggregator_args: Union[Dict, None]):
    """
    Initialize appropriate aggregator by type.
    """
    # TODO: Update as we add more aggregator types
    aggregator_mapping = {
        'concat': Concat,
        'global_attention': GlobalAttention,
        'gc': GoalConditioned,
        'pgp': PGP
    }

    if aggregator_args:
        return aggregator_mapping[aggregator_type](aggregator_args)
    else:
        return aggregator_mapping[aggregator_type]()


def initialize_decoder(decoder_type: str, decoder_args: Dict):
    """
    Initialize appropriate decoder by type.
    """
    # TODO: Update as we add more decoder types
    decoder_mapping = {
        'mtp': MTP,
        'multipath': Multipath,
        'covernet': CoverNet,
        'lvm': LVM
    }

    return decoder_mapping[decoder_type](decoder_args)


# Metrics
def initialize_metric(metric_type: str, metric_args: Dict = None):
    """
    Initialize appropriate metric by type.
    """
    # TODO: Update as we add more metrics
    metric_mapping = {
        'mtp_loss': MTPLoss,
        'covernet_loss': CoverNetLoss,
        'min_ade_k': MinADEK,
        'min_fde_k': MinFDEK,
        'miss_rate_k': MissRateK,
        'pi_bc': PiBehaviorCloning,
        'goal_pred_nll': GoalPredictionNLL
    }

    if metric_args is not None:
        return metric_mapping[metric_type](metric_args)
    else:
        return metric_mapping[metric_type]()
