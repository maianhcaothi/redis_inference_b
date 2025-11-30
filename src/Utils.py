import yaml

def load_config(path="./cfg/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_message(size_MB):
    return b"A"*(size_MB*1024*1024)