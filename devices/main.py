import threading
import time

from edge_device.edge_device import EdgeDevice
from middleware.connection_manager import ConnectionManager
from middleware.middleware import MiddleWare
from utils.utils import get_config_file_path, read_yaml


def start_Device(deviceName, accountNr, federation_mgr, config_file):
    edgeDevice = EdgeDevice(deviceName, config_file=config_file)
    # publisher loop runs inside EdgeDevice.start_EdgeDevice
    pub_thread = threading.Thread(
        target=edgeDevice.start_EdgeDevice,
        daemon=True  # allow main process to exit once work is done
    )
    pub_thread.start()

    middleware = MiddleWare(
        connection_manager=federation_mgr,
        deviceName=deviceName,
        accountNR=accountNr,
        configFile=config_file,
    )
    middleware.start_Middleware()


if __name__ == "__main__":
    _config_file_path = get_config_file_path()
    config_file = read_yaml(_config_file_path)
    participant_count = config_file["DEFAULT"]["NumberOfParticipants"]

    barrier = threading.Barrier(participant_count)

    # off-chain federation manager
    federation_mgr = ConnectionManager(
        config_file=config_file,
        participant_count=participant_count,
        barrier=barrier,
    )

    # launch all device threads and wait for completion
    threads = []
    for i in range(participant_count):
        t = threading.Thread(
            target=start_Device,
            args=[f"Device_{i+1}", i, federation_mgr, config_file],
            daemon=False
        )
        threads.append(t)
        t.start()
        time.sleep(1)

    # wait until all devices finish their federated rounds
    for t in threads:
        t.join()

    print("All devices completed. Exiting.")