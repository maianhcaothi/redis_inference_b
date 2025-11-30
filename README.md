# Redis Inference Benchmark

## Introduction

This repository benchmarks and compares three popular message brokers:
- **RabbitMQ** - Robust message queue with advanced routing
- **Redis** - In-memory data store with pub/sub capabilities
- **Kafka** - Distributed event streaming platform

The goal is to evaluate performance metrics and identify the best broker for your network requirements.

## Quick Start

### Prerequisites

Before running the benchmark, ensure you have set up:**RabbitMQ** , **Redis** and **Kafka** .

Then install Python dependencies:
```bash
pip install -r requirements.txt
```

### Running the Benchmark

**Start the Sender:**
```bash
python main.py --role sender
```

**Start the Receiver (in another terminal):**
```bash
python main.py --role receiver
```

The sender and receiver will communicate through the configured broker, measuring throughput and latency.

## Configuration

Edit `cfg/config.yaml` to customize the benchmark:

```yaml
# Broker Configuration
rabbit:
  address: "localhost"
  username: "guest"
  password: "guest"
  virtual-host: "/"

redis:
  host: "localhost"
  port: 6379
  db: 0

# Benchmark Settings
num_rounds: 100          # Number of messages to send
message_size: 15         # Message size in MB (max 16 MB)
```

## Project Structure

```
├── main.py              # Entry point (RabbitMQ, Redis, Kafka runners)
├── client.py            # Client utilities
├── src/
│   ├── Utils.py        # Configuration loader
│   ├── rabbitmq/       # RabbitMQ sender/receiver
│   ├── redis/          # Redis sender/receiver
│   └── kafka/          # Kafka sender/receiver
├── cfg/
│   └── config.yaml     # Broker configuration
│ 
└── requirements.txt    # Python dependencies
```

## Metrics

The benchmark measures:
- Message throughput (messages/sec)
- Latency (milliseconds)
- Resource utilization

## Notes

- Maximum message size: 15 MB
- Default rounds: 100 rounds
- Ensure all brokers are running before starting the benchmark
