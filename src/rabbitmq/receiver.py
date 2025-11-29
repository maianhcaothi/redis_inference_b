import pika
import pickle
from src.Utils import load_config


class Receiver:

    def __init__(self, config):
        rabbit = config["rabbit"]

        self.ready_queue = config["queues"]["ready"]
        self.ping_queue  = config["queues"]["ping"]
        self.pong_queue  = config["queues"]["pong"]

        self.num_rounds = config["num_rounds"]

        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=rabbit["address"],
                credentials=pika.PlainCredentials(
                    rabbit["username"],
                    rabbit["password"]
                ),
                virtual_host=rabbit["virtual-host"]
            )
        )

        self.channel = self.connection.channel()

        for q in [self.ready_queue, self.ping_queue, self.pong_queue]:
            self.channel.queue_declare(queue=q, durable=True)

    def run(self):
        print("Receiver started — sending READY signal")

        # Signal sender we are ready
        self.channel.basic_publish(
            exchange="",
            routing_key=self.ready_queue,
            body=pickle.dumps({"signal": "READY"})
        )

        print(f"Waiting for {self.num_rounds} pings...")

        received = 0

        while received < self.num_rounds:
            _, _, body = self.channel.basic_get(
                queue=self.ping_queue,
                auto_ack=True
            )

            if body:
                data = pickle.loads(body)

                # Echo back
                self.channel.basic_publish(
                    exchange="",
                    routing_key=self.pong_queue,
                    body=pickle.dumps(data)
                )

                received += 1
                # print(f"Ping {received}/{self.num_rounds} echoed")

        print("All pings processed")

    def clean(self):
        try:
            if self.channel and self.channel.is_open:
                self.channel.close()
            if self.connection and self.connection.is_open:
                self.connection.close()
            print("Receiver cleaned")
        except Exception as e:
            print("Cleanup error:", e)
