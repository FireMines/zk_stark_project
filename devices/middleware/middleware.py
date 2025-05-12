import functools
import io
import threading
import time

import pandas as pd
from analytics.analytics import Analytics
from message_broker.consumer import Consumer

from middleware.neural_net import FCLayer, Network, mse, mse_prime
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


def callback(ch, method, properties, body, model):
    """
    Pika consumer callback: parses the incoming CSV batch and
    forwards it to the FederatedLearningModel instance.
    """
    batch = pd.read_csv(io.BytesIO(body), header=0, index_col=0)
    print(f"{model.deviceName}: received batch with shape {batch.shape}")
    model.add_data_to_current_batch(batch)


class FederatedLearningModel:
    def __init__(self, config_file, deviceName):
        # Basic setup
        self.deviceName = deviceName
        self.config = config_file
        self.scaler = StandardScaler()
        self.net = Network(
            self.config["DEFAULT"]["OutputDimension"],
            self.config["DEFAULT"]["InputDimension"],
            self.config["DEFAULT"]["Precision"],
        )
        # Single-layer network
        self.net.add(
            FCLayer(
                self.config["DEFAULT"]["InputDimension"],
                self.config["DEFAULT"]["OutputDimension"],
            )
        )
        self.epochs = self.config["DEFAULT"]["Epochs"]
        self.net.use(mse, mse_prime)

        # Placeholders for local training
        self.learning_rate = None
        self.curr_batch = None
        self.batchSize = None

        # Load & preprocess test data
        datasource = self.config["DEFAULT"]["TestFilePath"]
        testdata = pd.read_csv(
            datasource,
            names=[
                "T_xacc","T_yacc","T_zacc","T_xgyro","T_ygyro","T_zgyro",
                "T_xmag","T_ymag","T_zmag","RA_xacc","RA_yacc","RA_zacc",
                "RA_xgyro","RA_ygyro","RA_zgyro","RA_xmag","RA_ymag","RA_zmag",
                "LA_xacc","LA_yacc","LA_zacc","LA_xgyro","LA_ygyro","LA_zgyro",
                "LA_xmag","LA_ymag","LA_zmag","RL_xacc","RL_yacc","RL_zacc",
                "RL_xgyro","RL_ygyro","RL_zgyro","RL_xmag","RL_ymag","RL_zmag",
                "LL_xacc","LL_yacc","LL_zacc","LL_xgyro","LL_ygyro","LL_zgyro",
                "LL_xmag","LL_ymag","LL_zmag","Activity",
            ],
        )
        testdata.fillna(method="backfill", inplace=True)
        testdata.dropna(inplace=True)
        testdata.drop(
            columns=[
                "T_xacc","T_yacc","T_zacc","T_xgyro","T_ygyro","T_zgyro",
                "T_xmag","T_ymag","T_zmag","RA_xacc","RA_yacc","RA_zacc",
                "RA_xgyro","RA_ygyro","RA_zgyro","RA_xmag","RA_ymag","RA_zmag",
                "RL_xacc","RL_yacc","RL_zacc","RL_xgyro","RL_ygyro","RL_zgyro",
                "RL_xmag","RL_ymag","RL_zmag","LL_xacc","LL_yacc","LL_zacc",
                "LL_xgyro","LL_ygyro","LL_zgyro","LL_xmag","LL_ymag","LL_zmag",
            ],
            inplace=True,
        )
        # Map & filter activities
        activity_mapping = self.config["DEFAULT"]["ActivityMappings"]
        filtered_activities = self.config["DEFAULT"]["Activities"]
        activity_encoding = self.config["DEFAULT"]["ActivityEncoding"]
        for key in activity_mapping:
            testdata.loc[testdata["Activity"] == key, "Activity"] = activity_mapping[key]
        testdata = testdata[testdata["Activity"].isin(filtered_activities)]
        for key in activity_encoding:
            testdata.loc[testdata["Activity"] == key, "Activity"] = activity_encoding[key]

        self.x_test = testdata.drop(columns="Activity")
        self.y_test = testdata["Activity"]

    def test_model(self):
        X = self.scaler.transform(self.x_test.to_numpy())
        preds = self.net.predict(X)
        return accuracy_score(self.y_test, preds)

    def get_classification_report(self):
        X = self.scaler.transform(self.x_test.to_numpy())
        return classification_report(
            self.y_test,
            self.net.predict(X),
            zero_division=0,
            output_dict=True,
        )

    def process_Batch(self):
        # Train on one batch
        self.curr_batch.dropna(inplace=True)
        batch = self.curr_batch.sample(self.batchSize)
        X_train = batch.drop(columns=self.config["DEFAULT"]["ResponseVariable"]).to_numpy()
        y_train = batch[self.config["DEFAULT"]["ResponseVariable"]].to_numpy()
        self.scaler.fit(self.x_test.to_numpy())
        X_scaled = self.scaler.transform(X_train)
        self.net.fit(
            X_scaled,
            y_train,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
        )
        print(f"{self.deviceName}: Score:", self.test_model())

    def reset_batch(self):
        self.curr_batch = None

    def get_weights(self):
        return self.net.get_weights()

    def get_bias(self):
        return self.net.get_bias()

    def set_learning_rate(self, rate):
        self.learning_rate = rate

    def set_weights(self, w):
        self.net.set_weights(w)

    def set_bias(self, b):
        self.net.set_bias(b)

    def set_batchSize(self, bs):
        self.batchSize = bs

    def set_precision(self, prec):
        self.net.set_precision(prec)

    def add_data_to_current_batch(self, data: pd.DataFrame):
        if self.curr_batch is None:
            self.curr_batch = data
        else:
            self.curr_batch = pd.concat([self.curr_batch, data])


class MiddleWare:
    def __init__(self, connection_manager, deviceName, accountNR, configFile):
        self.accountNR = accountNR
        self.connection_manager = connection_manager
        self.deviceName = deviceName
        self.config = configFile

        # Core model + analytics
        self.model = FederatedLearningModel(configFile, deviceName)
        self.analytics = Analytics(deviceName=deviceName, config_file=configFile)

        # Set up RabbitMQ consumer
        self.consumer = Consumer()
        on_message_callback = functools.partial(callback, model=self.model)
        queueName = self.config["DEFAULT"]["QueueBase"] + deviceName
        self.consumer.declare_queue(queueName)
        self.consumer.consume_data(queueName, on_message_callback)

        # Run the consumer in a background daemon thread
        self.consumer_thread = threading.Thread(
            target=self.consumer.start_consuming,
            daemon=True
        )

    def __start_Consuming(self):
        self.consumer_thread.start()

    def start_Middleware(self):
        self.__start_Consuming()
        self.round = 0
        print(f"{self.deviceName}: will run for {self.config['DEFAULT']['Rounds']} rounds")

        # Main federated-learning loop
        while self.round < self.config["DEFAULT"]["Rounds"]:
            print(f"{self.deviceName}: round {self.round} — checking for outstanding update…")
            if self.connection_manager.roundUpdateOutstanding(self.accountNR):
                print(f"{self.deviceName}: round {self.round} — outstanding_update = True")

                # Fetch latest off-chain global model
                global_w = self.connection_manager.get_globalWeights(self.accountNR)
                global_b = self.connection_manager.get_globalBias(self.accountNR)
                lr = self.connection_manager.get_LearningRate(self.accountNR)
                prec = self.connection_manager.get_Precision(self.accountNR)
                bs = self.connection_manager.get_BatchSize(self.accountNR)

                # Inject into local model
                self.model.set_precision(prec)
                self.model.set_learning_rate(lr)
                self.model.set_weights(global_w)
                self.model.set_bias(global_b)
                self.model.set_batchSize(bs)

                # Wait up to 10s for enough samples
                start_t = time.time()
                while (
                    (self.model.curr_batch is None or self.model.curr_batch.size < bs)
                    and time.time() - start_t < 10.0
                ):
                    time.sleep(0.1)

                if self.model.curr_batch is None or self.model.curr_batch.size < bs:
                    print(f"{self.deviceName}: timed out waiting for data, skipping round {self.round}")
                else:
                    # Local training + analytics
                    t0 = time.time()
                    self.model.process_Batch()
                    self.analytics.add_round_training_local_time(self.round, time.time() - t0)
                    self.analytics.add_round_score(self.round, self.model.test_model())
                    self.analytics.add_round_classification_report(
                        self.round,
                        self.model.get_classification_report(),
                    )

                    # Off-chain “update”
                    mse_score = self.model.net.mse_average
                    t1 = time.time()
                    self.connection_manager.update(
                        self.model.get_weights(),
                        self.model.get_bias(),
                        mse_score,
                        self.accountNR,
                    )
                    self.analytics.add_round_update_blockchain_time(self.round, time.time() - t1)

                self.round += 1

            time.sleep(self.config["DEFAULT"]["WaitingTime"])

        # Done: write analytics, tear down consumer
        self.analytics.write_data()
        print("Done.")
        try:
            self.consumer.channel.stop_consuming()
            self.consumer.connection.close()
        except Exception:
            pass
