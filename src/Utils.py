import yaml , os , csv

def load_config(path="./cfg/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_message(size_MB):
    return b"A"*(size_MB*1024*1024)


def append_csv(file_path, row  ):
    """
    Always append one row to CSV safely.
    Header is written once if file doesn't exist.

    Args:
        file_path (str)
        row (list or tuple)
    """
    header = []
    for i in range(1, 16):
        header.append(f"{i} MB")

    write_header = not os.path.exists(file_path)

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(header)

        writer.writerow(row)

    print("Saved to csv successfully ! ")
