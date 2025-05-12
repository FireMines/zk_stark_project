from typing import Optional
from middleware.aggregator import OffChainAggregator

class AggregatorSelector:
    """
    Off-chain selector: round-robin among aggregators in memory.
    """
    def __init__(self, connection_manager, aggregators: list[OffChainAggregator]):
        self.connection_manager = connection_manager
        self.aggregators = aggregators
        self._selected_aggregator = None
        self.is_initialized = False
        self.select()

    def select(self) -> None:
        if not self.is_initialized or self._selected_aggregator is None:
            idx = 0
        else:
            idx = (self.aggregators.index(self._selected_aggregator) + 1) % len(self.aggregators)
        self._selected_aggregator = self.aggregators[idx]
        self.is_initialized = True

    def store_device_wb(self, device_id, w, b, mse_score):
        return self._selected_aggregator.store_device_wb(device_id, w, b, mse_score)

    def start_round(self):
        return self._selected_aggregator.start_round()

    def finish_round(self):
        res = self._selected_aggregator.finish_round()
        self.select()
        return res

    def get_agg_obj_from_address(self, name: str) -> Optional[OffChainAggregator]:
        for agg in self.aggregators:
            if agg.name == name:
                return agg
        return None
