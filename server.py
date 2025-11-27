import argparse
import sys
import signal
from src.Server import Server

import src.Log
import yaml

parser = argparse.ArgumentParser(description="Split learning framework with controller.")
args = parser.parse_args()

with open('config.yaml') as file:
    config = yaml.safe_load(file)



def signal_handler(sig, frame):
    print("\nCatch stop signal Ctrl+C. Stop the program.")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    server = Server(config)
    server.start()
    src.Log.print_with_color("Ok, ready!", "green")
