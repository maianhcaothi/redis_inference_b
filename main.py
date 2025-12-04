import argparse

from src.Utils import load_config
from src.rabbitmq.sender import RabbitMQSender
from src.rabbitmq.receiver import RabbitMQReceiver
from src.redis.sender import RedisSender
from src.redis.receiver import RedisReceiver


def rabbit_mq():
    parser = argparse.ArgumentParser("RabbitMQ Runner")

    parser.add_argument(
        "--role",
        choices=["sender", "receiver"],
        required=True,
        help="Choose which component to run",
    )

    args, unknown = parser.parse_known_args()
    config = load_config()

    app = None
    if args.role == "sender":
        print("▶ Running RABBITMQ SENDER")
        app = RabbitMQSender(config) 

    elif args.role == "receiver":
        print("▶ Running RABBITMQ RECEIVER")
        app = RabbitMQReceiver(config) 
    
    if app:
        try:
            app.run()
        finally:
            app.clean()


def redis(): 
    parser = argparse.ArgumentParser("Redis")

    parser.add_argument(
        "--role",
        choices=["sender", "receiver"],
        required=True,
        help="Choose which component to run",
    )

    args, unknown = parser.parse_known_args()
    config = load_config()

    app = None
    if args.role == "sender":
        print("▶ Running REDIS SENDER")
        app = RedisSender(config) 

    elif args.role == "receiver":
        print("▶ Running REDIS RECEIVER")
        app = RedisReceiver(config) 
    
    if app:
        try:
            app.run()
        finally:
            app.clean()

if __name__ == "__main__":
    #rabbit_mq()
    redis()
    # kafka()
