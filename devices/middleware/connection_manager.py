import copy
import threading
import numpy as np

from middleware.aggregator import OffChainAggregator
from middleware.aggregator_selection import AggregatorSelector
from middleware.ipfs import IPFSConnector


class ConnectionManager:
    def __init__(self, config_file, participant_count: int, barrier: threading.Barrier):
        self.config = config_file
        self.participant_count = participant_count
        self.barrier = barrier

        # stubbed IPFS connector
        self.ipfs = IPFSConnector()

        # initialize global model off-chain
        np.random.seed(4)
        W = (
            np.random.randn(
                config_file["DEFAULT"]["OutputDimension"],
                config_file["DEFAULT"]["InputDimension"],
            )
            * config_file["DEFAULT"]["Precision"]
            / 5
        )
        b = (
            np.random.randn(config_file["DEFAULT"]["OutputDimension"])
            * config_file["DEFAULT"]["Precision"]
            / 5
        )
        self.global_w = [[int(x) for x in row] for row in W]
        self.global_b = [int(x) for x in b]

        # pretend IPFS links
        self.weight_ipfs_link = self.ipfs.save_global_weight(self.global_w)
        self.bias_ipfs_link = self.ipfs.save_global_bias(self.global_b)

        # two off-chain aggregators
        aggs = [
            OffChainAggregator("FirstAgg", self, self.ipfs, self.global_w, self.global_b),
            OffChainAggregator("SecondAgg", self, self.ipfs, self.global_w, self.global_b),
        ]
        self.aggregator_selector = AggregatorSelector(self, aggs)

        self.lock_newRound = threading.Lock()

    def get_LearningRate(self, _):
        return self.config["DEFAULT"]["LearningRate"]

    def get_BatchSize(self, _):
        return self.config["DEFAULT"]["BatchSize"]

    def get_Precision(self, _):
        return self.config["DEFAULT"]["Precision"]

    def get_globalWeights(self, _):
        return copy.deepcopy(self.global_w)

    def get_globalBias(self, _):
        return copy.deepcopy(self.global_b)

    def roundUpdateOutstanding(self, _):
        with self.lock_newRound:
            return True

    def update(self, weights, bias, mse_score, device_id, proof=None):
        self.aggregator_selector.store_device_wb(
            device_id=device_id,
            w=weights,
            b=bias,
            mse_score=mse_score,
        )
