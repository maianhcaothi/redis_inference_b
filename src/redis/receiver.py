import redis
import pickle
import time
from src.Utils import load_config


class RedisReceiver:

    def __init__(self, config):
        redis_config = config["redis"]

        self.ready_queue = config["queues"]["ready"]
        self.ping_queue  = config["queues"]["ping"]
        self.pong_queue  = config["queues"]["pong"]
        self.num_rounds = config["num_rounds"]

        self.r = redis.StrictRedis(
            host=redis_config["host"],
            port=redis_config["port"],
            password=redis_config.get("password"),
            decode_responses=False
        )
        print("Redis Receiver initialized")

    def run(self):
        print("Receiver started — sending READY signal")

        # Signal sender we are ready
        self.r.rpush(
            self.ready_queue,
            pickle.dumps({"signal": "READY"})
        )

        print(f"Waiting for {self.num_rounds} pings...")

        received = 0

        while received < self.num_rounds:
            item = self.r.blpop(self.ping_queue, timeout=1)

            if item:
                _,body = item
                data = pickle.loads(body)

                # Echo back
                self.r.rpush(
                    self.pong_queue,
                    pickle.dumps(data)
                )

                received += 1
                # print(f"Ping {received}/{self.num_rounds} echoed")

        print("All pings processed")

    def clean(self):
        try:
            print("Receiver cleaned (Redis connection closed if necessary)")
        except Exception as e:
            print("Cleanup error:", e)

