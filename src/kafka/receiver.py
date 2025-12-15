import time
import pickle
from kafka import KafkaConsumer, KafkaProducer
from src.Utils import load_config

class  KafkaReceiver:
    def __init__(self, config):
        kafka_config = config["kafka"] 

        self.ready_topic = config["queues"]["ready"]
        self.ping_topic  = config["queues"]["ping"]
        self.pong_topic  = config["queues"]["pong"]

        self.num_rounds = config["num_rounds"]
        
        # Logic đếm số lần chạy dựa trên message_size
        self.cnt = config["message_size"]
        if self.cnt == 0:
            self.cnt = 15 # Lặp 15 lần (từ 1MB đến 15MB)
        else:
            self.cnt = 1  # Chỉ chạy 1 lần với kích thước cố định

        # Cấu hình Kafka Broker Address
        self.bootstrap_servers = [f"{kafka_config['address']}:{kafka_config['port']}"]

        # 1. Khởi tạo Producer (để gửi READY và PONG)
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=pickle.dumps, 
            acks='all'
        )

        # 2. Khởi tạo Consumer (để nhận PING - gói tin lớn)
        self.consumer = KafkaConsumer(
            self.ping_topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id='kafka-rtt-receiver-group', 
            auto_offset_reset='latest', # Chỉ đọc các tin nhắn mới
            value_deserializer=pickle.loads,
            max_partition_fetch_bytes=20971520,
            consumer_timeout_ms=500 # Giới hạn thời gian chờ poll
        )
        print("Kafka Receiver initialized")

    def run(self):
        # Vòng lặp chính để xử lý các kích thước tải trọng khác nhau (hoặc chỉ một lần)
        while self.cnt > 0:
            self.cnt -= 1
            print("Receiver started — sending READY signal")

            # 1. Gửi tín hiệu READY (tương đương RPUSH)
            self.producer.send(
                self.ready_topic, 
                {"signal": "READY"}
            )
            self.producer.flush() # Đảm bảo tin nhắn được gửi đi

            print(f"Waiting for {self.num_rounds} pings...")
            
            received = 0
            
            # 2. Vòng lặp chờ PING (tương đương BLPOP loop)
            while received < self.num_rounds:
                # Poll để nhận các tin nhắn Ping
                # Kafka Consumer không có BLPOP, phải dùng poll()
                msg_pack = self.consumer.poll(timeout_ms=100) 
                
                if msg_pack:
                    for _, messages in msg_pack.items():
                        for message in messages:
                            # Đảm bảo tin nhắn là từ chủ đề ping
                            if message.topic == self.ping_topic:
                                if received >= self.num_rounds:
                                    break
                                
                                data = message.value # Gói dữ liệu lớn đã được giải tuần tự hóa
                                
                                # 3. Gửi lại Pong (Echoing the payload)
                                self.producer.send(
                                    self.pong_topic, 
                                    data
                                )
                                self.producer.flush() 

                                received += 1
                                # print(f"Ping {received}/{self.num_rounds} echoed")
                
                # Thoát vòng lặp bên trong nếu đã đủ số lần nhận
                if received >= self.num_rounds:
                    break

        print("All pings processed")

    def clean(self):
        try:
            if self.producer:
                self.producer.close()
            if self.consumer:
                self.consumer.close()
            print("Receiver cleaned (Kafka connections closed)")
        except Exception as e:
            print("Cleanup error:", e)