import redis
import pickle
import time
from statistics import mean
from src.Utils import load_config,  get_message , append_csv

class RedisSender:
    def __init__(self, config):
        redis_config = config["redis"]

        self.ready_queue = config["queues"]["ready"]
        self.ping_queue  = config["queues"]["ping"]
        self.pong_queue  = config["queues"]["pong"]

        self.num_rounds = config.get("num_rounds", 100)
        self.size_MB = config["message_size"]
        # self.message = get_message(self.size_MB)
        self.lst_data = []

        self.r = redis.StrictRedis(
            host=redis_config["host"],
            port=redis_config["port"],
            password=redis_config.get("password"),
            decode_responses=False
        )
        print("Redis Sender initialized")

    def wait_for_receiver(self):
        print("Waiting for receiver to become ready...")

        while True:
            item = self.r.blpop(self.ready_queue, timeout=1)
            if item:
                print("Receiver is READY ")
                return

    def measure_round_trip(self):
        message = {
        "signal": get_message(self.size_MB)
    }

        start_ns = time.time_ns()

        self.r.rpush(
            self.ping_queue,
            pickle.dumps(message)
        )

        # wait for pong
        while True:
            item = self.r.blpop(self.pong_queue, timeout=1)
            if item:
                end_ns = time.time_ns()
                break

        rtt_ms = (end_ns - start_ns) / 1e6
        return rtt_ms

    def single_run(self):
        self.wait_for_receiver()

        print(f"\nRunning {self.num_rounds} RTT rounds...\n")
        print(f"[Size MB] {self.size_MB}")

        times = []

        for i in range(self.num_rounds):
            rtt = self.measure_round_trip()
            times.append(rtt)
            # print(f"Round {i+1:4}/{self.num_rounds} : RTT = {rtt:.3f} ms")

        # print("\n===== RESULTS =====")
        # print(f"Avg RTT   : {mean(times):.3f} ms")
        # print(f"Min RTT   : {min(times):.3f} ms")
        # print(f"Max RTT   : {max(times):.3f} ms")

        # P95 RTT calculation is fine, ensure sorted() handles data correctly
        # print(f"P95 RTT   : {times[p95_index]:.3f} ms")

        # approximate one-way transfer time
        print(f"~One-way transfer time ≈ {mean(times)/2:.3f} ms")
        self.lst_data.append(f"{mean(times)/2:.3f} ms")

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

            print("Sender cleaned (Redis connection closed if necessary) ")
        except Exception as e:
            print("Cleanup error:", e)
