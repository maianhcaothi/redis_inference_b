import yaml

def load_config(path="./cfg/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# load_config()