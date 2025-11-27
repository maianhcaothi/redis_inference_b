import numpy as np


def quantize(x: np.ndarray, bits: int = 8):

    x_min, x_max = x.min(), x.max()
    scale = (x_max - x_min) / (2 ** bits - 1)
    q = np.round((x - x_min) / scale).astype(np.uint8 if bits <= 8 else np.uint16)
    return q, x_min, scale


def dequantize(q: np.ndarray, x_min: float, scale: float):
    return q.astype(np.float32) * scale + x_min


# ====== TẠO SƠ ĐỒ PHÂN TẦNG ======
def make_layer_map(C: int, scheme="8+8+8+8+8+8+8"):
    """
    Trả về list các chỉ số kênh cho từng layer theo chuỗi `scheme`, ví dụ "1+1+1+1+4+4"
    nghĩa là 1 kênh/tầng cho 4 tầng đầu + 4 kênh/tầng cho 2 tầng cao.
    """
    nums = [int(x) for x in scheme.split("+")]
    assert sum(nums) <= C, "Số kênh > C"
    idx = np.arange(C)
    start = 0
    groups = []
    for n in nums:
        groups.append(idx[start:start + n])
        start += n
    if start < C:  # kênh dư -> tầng cuối
        groups.append(idx[start:])
    return groups


# ====== NÉN ======
def compress_layers(x, layer_map, bits_per_layer):
    """
    x: tensor (N,C,H,W) float32
    layer_map: list np.ndarray indices cho mỗi layer
    bits_per_layer: int or list
    """
    if isinstance(bits_per_layer, int):
        bits_per_layer = [bits_per_layer] * len(layer_map)

    streams, metas = [], []
    for idxs, b in zip(layer_map, bits_per_layer):
        part = x[:, idxs, ...]  # lấy kênh nhóm
        q, x_min, scale = quantize(part, b)
        streams.append(q)  # ở đây ta lưu mảng uint{8/16}; thực tế -> bytes entropy-coded
        metas.append({"idxs": idxs, "min": x_min, "scale": scale, "bits": b})
    global_meta = {"C": x.shape[1], "shape": x.shape}
    return streams, metas, global_meta


# ====== GIẢI NÉN ======
def decompress_layers(streams, metas, global_meta, n_layers=None):
    """
    n_layers: chỉ dùng N tầng đầu (None -> tất cả)
    """
    if n_layers is None:
        n_layers = len(streams)
    C = global_meta["C"]
    N, _, H, W = global_meta["shape"]
    out = np.zeros((N, C, H, W), dtype=np.float32)

    for q, m in zip(streams[:n_layers], metas[:n_layers]):
        part = dequantize(q, m["min"], m["scale"])
        out[:, m["idxs"], ...] = part
    return out


def Encoder3(data_output, num_bits=8):
    layer_map = make_layer_map(data_output[-1].shape[1], "18+16+16+10+1+1+1+1")
    k = 6
    encoded_data = []
    meta_list = []
    global_meta_list = []
    for output in data_output:
        if output is None:
            encoded_data.append(None)
            meta_list.append(0)
            global_meta_list.append(0)
        else:
            streams, metas, gmeta = compress_layers(output, layer_map, bits_per_layer=num_bits)
            encoded_data.append(streams[:k])
            meta_list.append(metas[:k])
            global_meta_list.append(gmeta)

    return encoded_data, meta_list, global_meta_list

def Decoder3(encoded_data, meta_list, global_meta_list):
    decoded_data = []
    for i, encoded in enumerate(encoded_data):
        if encoded is None:
            decoded_data.append(None)
        else:
            metas = meta_list[i]
            gmeta = global_meta_list[i]
            decode = decompress_layers(encoded, metas, gmeta, n_layers=len(encoded))
            decoded_data.append(decode)
    return decoded_data



