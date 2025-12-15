import pika
import pickle
import time
from statistics import mean
from src.Utils import load_config, get_message , append_csv

class RabbitMQSender:
    def __init__(self, config):
        rabbit = config["rabbit"]

        self.ready_queue = config["queues"]["ready"]
        self.ping_queue  = config["queues"]["ping"]
        self.pong_queue  = config["queues"]["pong"]

        self.num_rounds = config.get("num_rounds", 100)
        self.size_MB = config["message_size"]
        # self.message = get_message(self.size_MB)
        self.lst_data = []

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

    def wait_for_receiver(self):
        print("Waiting for receiver to become ready...")

        while True:
            _, _, body = self.channel.basic_get(
                queue=self.ready_queue,
                auto_ack=True
            )
            if body:
                print("Receiver is READY ")
                return

    def measure_round_trip(self):
        message = {
            "signal": get_message(self.size_MB)
        }

        start_ns = time.time_ns()

        self.channel.basic_publish(
            exchange="",
            routing_key=self.ping_queue,
            body=pickle.dumps(message)
        )

        # wait for pong
        while True:
            _, _, body = self.channel.basic_get(
                queue=self.pong_queue,
                auto_ack=True
            )
            if body:
                end_ns = time.time_ns()
                break

        rtt_ms = (end_ns - start_ns) / 1e6
        return rtt_ms

    def single_run(self):
        self.wait_for_receiver()

        print(f"\nRunning {self.num_rounds} RTT rounds...\n")
        print(f"[Size MB] { self.size_MB}")

        times = []

        for i in range(self.num_rounds):
            rtt = self.measure_round_trip()
            times.append(rtt)
            # print(f"Round {i+1:4}/{self.num_rounds} : RTT = {rtt:.3f} ms")

        # print("\n===== RESULTS =====")
        # print(f"Avg RTT   : {mean(times):.3f} ms")
        # print(f"Min RTT   : {min(times):.3f} ms")
        # print(f"Max RTT   : {max(times):.3f} ms")
        # print(f"P95 RTT   : {sorted(times)[int(self.num_rounds*0.95)]:.3f} ms")

        # approximate one-way transfer time
        print(f"~One-way transfer time ≈ {mean(times)/2:.3f} ms")
        self.lst_data.append(f"{mean(times)/2:.3f}ms")


    def run(self):
        print(f"Message : {self.size_MB}")
        if self.size_MB != 0:
            self.single_run()
        else :
            while self.size_MB< 15 :
                self.size_MB += 1
                self.single_run()

        print(self.lst_data)
        append_csv("res.csv" , self.lst_data)



    def clean(self):
        try:
            if self.channel and self.channel.is_open:
                self.channel.close()
            if self.connection and self.connection.is_open:
                self.connection.close()
            print("Sender cleaned ")
        except Exception as e:
            print("Cleanup error:", e)
