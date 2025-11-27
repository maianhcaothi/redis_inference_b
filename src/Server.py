import os
import sys
import base64
import pickle
import torch
import torch.nn as nn
import redis

import src.Model
import src.Log
from ultralytics import YOLO
from src.redis_config import QUEUE_REGISTER_CLIENTS, QUEUE_FEATURE_MAP_PREFIX, BLOCKING_TIMEOUT

class Server:
    def __init__(self, config):
        self.model_name = config["server"]["model"]
        self.total_clients = config["server"]["clients"]
        self.cut_layer = config["server"]["cut-layer"]
        self.batch_frame = config["server"]["batch-frame"]

        redis_conf= config["redis"]
        self.redis_client = redis.Redis(
            host=redis_conf['address'],
            port=redis_conf['port'],
            password=redis_conf.get('password')
        )
        try:
            self.redis_client.ping()
        except Exception as e:
            src.Log.print_with_color(f"ERROR: Server failed to connect to Redis. {e}", "red")
            sys.exit(1)

        self.register_clients = [0 for _ in range(len(self.total_clients))]
        self.list_clients = []

        self.data = config["data"]
        self.debug_mode = config["debug-mode"]
        self.compress = config["compress"]
        self.cal_map = config["cal_map"]

        log_path = config["log-path"]
        self.logger = src.Log.Logger(f"{log_path}/app.log")
        self.logger.log_info(f"Application start. Server is waiting for {len(self.total_clients)} clients.")

    def on_request(self, message):
        action = message["action"]
        client_id = message["client_id"]
        layer_id = message["layer_id"]

        if action == "REGISTER":
            if (str(client_id), layer_id) not in self.list_clients:
                self.list_clients.append((str(client_id), layer_id))

            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            # Save messages from clients
            self.register_clients[layer_id-1] += 1

            # If consumed all clients - Register for first time
            if self.register_clients == self.total_clients:
                src.Log.print_with_color("All clients are connected. Sending notifications.", "green")
                self.notify_clients()


    def send_to_response(self, client_id, message):
        reply_key_name = f"reply_{client_id}"
        self.redis_client.set(reply_key_name, message, ex=600)
        src.Log.print_with_color(f"[>>>] Sent notification to client {client_id} via Redis key: {reply_key_name}", "red")


    def start(self):
        print(f"Server is actively listening for registration messages on key: {QUEUE_REGISTER_CLIENTS}")
        while True:
            item = self.redis_client.blpop(QUEUE_REGISTER_CLIENTS, timeout=BLOCKING_TIMEOUT)
            if item:
                body = item[1]
                message = pickle.loads(body)
                self.on_request(message)
            else:
                continue

    def notify_clients(self):
        default_splits = {
            "a": (4, [3]),
            "b": (11, [4, 6, 10]),
            "c": (17, [10, 13, 16]),
            "d": (23, [16, 19, 22])
        }
        model = YOLO(f"{self.model_name}.pt")
        splits = default_splits[self.cut_layer]
        file_path = f"{self.model_name}.pt"
        if os.path.exists(file_path):
            src.Log.print_with_color(f"Load model {self.model_name}.", "green")
            with open(f"{self.model_name}.pt", "rb") as f:
                file_bytes = f.read()
                encoded = base64.b64encode(file_bytes).decode('utf-8')
        else:
            src.Log.print_with_color(f"{self.model_name} does not exist.", "yellow")
            sys.exit()

        for (client_id, layer_id) in self.list_clients:

            response = {"action": "START",
                        "message": "Server accept the connection",
                        "model": encoded,
                        "splits": splits[0],
                        "save_layers": splits[1],
                        "batch_frame": self.batch_frame,
                        "num_layers": len(self.total_clients),
                        "model_name": self.model_name,
                        "data": self.data,
                        "debug_mode": self.debug_mode,
                        "compress": self.compress,
                        "cal_map": self.cal_map}
            
            self.send_to_response(client_id, pickle.dumps(response))
