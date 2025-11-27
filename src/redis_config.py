# src/redis_config.py

# Key dùng cho Server nhận yêu cầu đăng ký từ Client (RPC)
QUEUE_REGISTER_CLIENTS = 'split_rpc_register_queue'

# Tiền tố cho hàng đợi truyền Feature Maps giữa các lớp (Data Plane)
QUEUE_FEATURE_MAP_PREFIX = 'split_feature_map'

# Tên key dùng để gửi tín hiệu STOP/DỪNG
QUEUE_STOP_SIGNAL = 'split_stop_signal'

# Timeout Blocking cho BLPOP (tính bằng giây)
# Server/Client sẽ chờ 5s, nếu không có tin nhắn thì thử lại vòng lặp
BLOCKING_TIMEOUT = 5