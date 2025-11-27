import numpy as np

def random_mask(num_channels: int, keep: int, seed=None):
    """
    Sinh mặt nạ boolean dài `num_channels`, giữ lại `keep` kênh.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(num_channels, keep, replace=False)
    mask = np.zeros(num_channels, dtype=bool)
    mask[idx] = True
    return mask


def quantize(x: np.ndarray, bits: int = 8):
    """
    Lượng hoá tuyến tính sang `bits`-bit (symmetrical, unsigned).
    Trả về (tensor lượng hoá, min, scale) để đảo ngược.
    """
    x_min, x_max = x.min(), x.max()
    scale = (x_max - x_min) / (2**bits - 1)
    q = np.round((x - x_min) / scale).astype(np.uint8 if bits <= 8 else np.uint16)
    return q, x_min, scale


def dequantize(q, x_min, scale):
    return q.astype(np.float32) * scale + x_min


def compress(tensor: np.ndarray, mask: np.ndarray, bits: int = None):
    """
    Giả sử tensor shape = (N, C, H, W) hoặc (C, H, W)
    - mask: bool vector len=C (True = giữ)
    - bits: None  -> không lượng hoá; 8/4/... -> lượng hoá
    """
    orig_C = tensor.shape[-3]
    if tensor.ndim == 4:
        compressed = tensor[:, mask, ...]
    else:  # 3-D
        compressed = tensor[mask, ...]
    meta = {"mask": mask, "orig_C": orig_C}

    if bits is not None:
        q, x_min, scale = quantize(compressed, bits)
        meta.update({"quant": True, "bits": bits, "x_min": x_min, "scale": scale})
        return q, meta
    else:
        meta.update({"quant": False})
        return compressed, meta


def decompress(data, meta):
    mask = meta["mask"]
    C = meta["orig_C"]
    if meta["quant"]:
        data = dequantize(data, meta["x_min"], meta["scale"])

    # tái dựng tensor với kênh đã lược bỏ đặt = 0
    if data.ndim == 4:
        N, _, H, W = data.shape
        out = np.zeros((N, C, H, W), dtype=data.dtype)
        out[:, mask, ...] = data

    else:
        _, H, W = data.shape
        out = np.zeros((C, H, W), dtype=data.dtype)
        out[mask, ...] = data
    return out

def Encoder2(data_output):
    encoded_data = []
    list_meta = []
    for output in data_output:
        if output is None:
            encoded_data.append(None)
            list_meta.append(0)
        else:
            mask = random_mask(output.shape[1], keep=60, seed=42)
            encoded, meta = compress(output, mask, bits=8)

            encoded_data.append(encoded)
            list_meta.append(meta)

    return encoded_data, list_meta

def Decoder2(encoded_data, list_meta):
    decoded_data = []
    for i, encoded in enumerate(encoded_data):
        if encoded is None:
            decoded_data.append(None)
        else:
            meta = list_meta[i]
            decoded = decompress(encoded, meta)
            decoded_data.append(decoded)
    return decoded_data
