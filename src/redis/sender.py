import redis
import pickle
import time
from statistics import mean
from src.Utils import load_config,  get_message

class RedisSender:
    def __init__(self, config):
        redis_config = config["redis"]

        self.ready_queue = config["queues"]["ready"]
        self.ping_queue  = config["queues"]["ping"]
        self.pong_queue  = config["queues"]["pong"]

        self.num_rounds = config.get("num_rounds", 100)

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
        message = {"signal": "PING"}

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

    def run(self):
        self.wait_for_receiver()

        print(f"\nRunning {self.num_rounds} RTT rounds...\n")

        times = []

        for i in range(self.num_rounds):
            rtt = self.measure_round_trip()
            times.append(rtt)
            # print(f"Round {i+1:4}/{self.num_rounds} : RTT = {rtt:.3f} ms")

        print("\n===== RESULTS =====")
        print(f"Avg RTT   : {mean(times):.3f} ms")
        print(f"Min RTT   : {min(times):.3f} ms")
        print(f"Max RTT   : {max(times):.3f} ms")

        # P95 RTT calculation is fine, ensure sorted() handles data correctly
        times.sort()
        p95_index = int(self.num_rounds * 0.95)
        # Check if list is big enough, otherwise use last element
        p95_index = min(p95_index, len(times) - 1) 
        print(f"P95 RTT   : {times[p95_index]:.3f} ms")

        # approximate one-way transfer time
        print(f"~One-way transfer time ≈ {mean(times)/2:.3f} ms")

    def clean(self):
        try:

            print("Sender cleaned (Redis connection closed if necessary) ")
        except Exception as e:
            print("Cleanup error:", e)
