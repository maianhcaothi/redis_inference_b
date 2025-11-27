import uuid
import argparse
import yaml
import sys
import os
import pickle

import torch
import redis 

import src.Log
from src.RpcClient import RpcClient
from src.Scheduler import Scheduler

parser = argparse.ArgumentParser(description="Split learning framework")
parser.add_argument('--layer_id', type=int, required=True, help='ID of layer, start from 1')
parser.add_argument('--device', type=str, required=False, help='Device of client')

args = parser.parse_args()

try:
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    print("ERROR: config.yaml file not found.")
    sys.exit(1)
    
client_id = uuid.uuid4()


device = None

if args.device is None:
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using device: {torch.cuda.get_device_name(device)}")
    else:
        device = "cpu"
        print(f"Using device: CPU")
else:
    device = args.device
    print(f"Using device: {device}")


if __name__ == "__main__":
    src.Log.print_with_color("[>>>] Client sending registration message to server...", "red")
    data = {"action": "REGISTER", "client_id": client_id, "layer_id": args.layer_id, "message": "Hello from Client!"}
    try:
        message_bytes = pickle.dumps(data)
    except pickle.PickleError as e:
        print(f"ERROR: Failed to serialize registration data. {e}")
        sys.exit(1)
        
    scheduler = Scheduler(client_id, args.layer_id, config, device)
    client = RpcClient(client_id, args.layer_id, config, scheduler.inference_func, scheduler.check_compress_func, device)
    client.send_to_server(message_bytes)
    client.wait_response()
