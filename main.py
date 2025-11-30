import argparse

from src.Utils import load_config
from src.rabbitmq.sender import Sender
from src.rabbitmq.receiver import Receiver


def rabbit_mq():
    parser = argparse.ArgumentParser("RabbitMQ Runner")

    parser.add_argument(
        "--role",
        choices=["sender", "receiver"],
        required=True,
        help="Choose which component to run",
    )

    args = parser.parse_args()
    config = load_config()

    if args.role == "sender":
        print("▶ Running SENDER")
        app = Sender(config)

    elif args.role == "receiver":
        print("▶ Running RECEIVER")
        app = Receiver(config)

    try:
        app.run()
    finally:
        app.clean()


if __name__ == "__main__":
    rabbit_mq()
    # redis()
    # kafka()
