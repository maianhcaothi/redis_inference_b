import time
import pickle
from statistics import mean
from kafka import KafkaProducer, KafkaConsumer
from src.Utils import load_config, get_message, append_csv 

class KafkaSender:
    def __init__(self, config):
        kafka_config = config["kafka"] 

        self.ready_topic = config["queues"]["ready"]
        self.ping_topic  = config["queues"]["ping"]
        self.pong_topic  = config["queues"]["pong"]

        self.num_rounds = config.get("num_rounds", 100)
        self.size_MB = config["message_size"]
        self.lst_data = []

        # Cấu hình Kafka Broker Address
        self.bootstrap_servers = [f"{kafka_config['address']}:{kafka_config['port']}"]

        # 1. Khởi tạo Producer (Để gửi PING và READY)
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=pickle.dumps,
            acks='all' # Đảm bảo độ bền cho việc gửi tin
            max_request_size=20971520
        )

        # 2. Khởi tạo Consumer (Để nhận PONG và READY signal)
        # Sử dụng Group ID duy nhất cho mỗi lần chạy để tránh xung đột offset
        self.consumer = KafkaConsumer(
            self.ready_topic,
            self.pong_topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=f'kafka-rtt-sender-{time.time_ns()}', 
            auto_offset_reset='latest', # Bắt đầu đọc từ tin nhắn mới nhất
            value_deserializer=pickle.loads,
            max_partition_fetch_bytes=20971520,
            consumer_timeout_ms=500 # Giới hạn thời gian chờ poll
        )
        print("Kafka Sender initialized")

    def wait_for_receiver(self):
        print("Waiting for receiver to become ready...")
        
        # Chờ tin nhắn READY từ Receiver
        for message in self.consumer:
            if message.topic == self.ready_topic and message.value.get("signal") == "READY":
                print("Receiver is READY ")
                return

    def measure_round_trip(self):
        message = {"signal": get_message(self.size_MB)}
        start_ns = time.time_ns()

        self.producer.send(self.ping_topic, message)
        self.producer.flush() 

        # 2. Chờ PONG (Vòng lặp Poll tối ưu hơn)
        while True:
            # Poll với timeout 500ms (đã cấu hình ở trên)
            msg_pack = self.consumer.poll(timeout_ms=500) 
            
            if msg_pack:
                for _, messages in msg_pack.items():
                    for msg in messages:
                        if msg.topic == self.pong_topic:
                            end_ns = time.time_ns()
                            return (end_ns - start_ns) / 1e6
                            
    def single_run(self):
        self.wait_for_receiver()

        print(f"\nRunning {self.num_rounds} RTT rounds...\n")
        print(f"[Size MB] {self.size_MB}")

        times = []

        for i in range(self.num_rounds):
            rtt = self.measure_round_trip()
            times.append(rtt)
        
        # Tính toán và in kết quả
        print(f"~One-way transfer time ≈ {mean(times)/2:.3f} ms")
        self.lst_data.append(f"{mean(times)/2:.3f} ms")

    def run(self):
        print(f"Message : {self.size_MB}")
        
        # Logic lặp lại kích thước bản tin
        if self.size_MB != 0:
            self.single_run()
        else :
            current_size = 0 
            while current_size < 15 :
                current_size += 1
                self.size_MB = current_size # Cập nhật kích thước cho vòng lặp
                self.single_run() 

        print(self.lst_data)
        append_csv("res.csv" , self.lst_data)
        
    def clean(self):
        try:
            if self.producer:
                self.producer.close()
            if self.consumer:
                self.consumer.close()
            print("Sender cleaned (Kafka connections closed) ")
        except Exception as e:
            print("Cleanup error:", e)