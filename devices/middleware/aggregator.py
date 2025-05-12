import copy

def moving_average_weights(local_weights, participant_count, global_weights):
    new_weights = [
        [0 for _ in range(len(global_weights[0]))]
        for _ in range(len(global_weights))
    ]
    k = participant_count
    for w in local_weights:
        for i in range(len(global_weights)):
            for j in range(len(global_weights[i])):
                delta = (w[i][j] - global_weights[i][j]) / k
                new_weights[i][j] += delta
    for i in range(len(global_weights)):
        for j in range(len(global_weights[i])):
            new_weights[i][j] += global_weights[i][j]
    return [[int(x) for x in row] for row in new_weights]


def moving_average_bias(local_bias, participant_count, global_bias):
    new_bias = [0 for _ in range(len(global_bias))]
    k = participant_count
    for b in local_bias:
        for i in range(len(global_bias)):
            delta = (b[i] - global_bias[i]) / k
            new_bias[i] += delta
    for i in range(len(global_bias)):
        new_bias[i] += global_bias[i]
    return [int(x) for x in new_bias]


class OffChainAggregator:
    def __init__(self, name, connection_manager, ipfs, global_w, global_b):
        self.name = name
        self.connection_manager = connection_manager
        self.ipfs = ipfs
        self.global_w = copy.deepcopy(global_w)
        self.global_b = copy.deepcopy(global_b)
        self.stored_device_data = {}

    def store_device_wb(self, device_id, w, b, mse_score):
        self.stored_device_data[device_id] = (w, b, mse_score)

    def start_round(self):
        self.global_w = copy.deepcopy(self.connection_manager.global_w)
        self.global_b = copy.deepcopy(self.connection_manager.global_b)
        self.stored_device_data.clear()

    def finish_round(self):
        if not self.stored_device_data:
            return
        local_ws = [v[0] for v in self.stored_device_data.values()]
        local_bs = [v[1] for v in self.stored_device_data.values()]
        new_w = moving_average_weights(local_ws, len(local_ws), self.global_w)
        new_b = moving_average_bias(local_bs, len(local_bs), self.global_b)
        self.connection_manager.global_w = new_w
        self.connection_manager.global_b = new_b
