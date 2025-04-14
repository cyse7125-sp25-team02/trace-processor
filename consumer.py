# trace-processor/consumer.py
import json
import logging
import os
import time
from confluent_kafka import Consumer, KafkaException
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KafkaConsumer:
    def __init__(self, bootstrap_servers: str, group_id: str, topic: str):
        self.config = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'lastest',
            'enable.auto.commit': True,
        }
        self.topic = topic
        self.consumer = None

    def connect(self) -> None:
        for attempt in range(5):
            try:
                self.consumer = Consumer(self.config)
                self.consumer.subscribe([self.topic])
                logger.info(f"Subscribed to topic: {self.topic}")
                return
            except KafkaException as e:
                logger.error(f"Connection attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)
        raise KafkaException("Failed to connect to Kafka after retries")

    def process_message(self, message: Dict) -> None:
        logger.info(f"Processing JSON message: {message}")
        # In phase 3, add PDF processing here

    def consume(self) -> None:
        if not self.consumer:
            raise ValueError("Consumer not connected. Call connect() first.")

        while True:
            try:
                msg = self.consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    logger.error(f"Consumer error: {msg.error()}")
                    continue
                try:
                    raw_message = msg.value()
                    logger.info(f"Raw message: {raw_message}")
                    if raw_message is None or len(raw_message) == 0:
                        logger.warning("Received empty or null message")
                        continue
                    message_value = raw_message.decode('utf-8')
                    try:
                        message = json.loads(message_value)
                        logger.info(f"Parsed JSON message: {message}")
                        self.process_message(message)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON message: {e}, raw: {message_value}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}, raw: {raw_message}")
            except KafkaException as e:
                logger.error(f"Kafka error: {e}. Reconnecting...")
                self.close()
                self.connect()

    def close(self) -> None:
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed")
            self.consumer = None

def main():
    # Read from environment variables with defaults
    bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka-cluster-dns-name:9092')
    group_id = os.getenv('KAFKA_GROUP_ID', 'group-name')
    topic = os.getenv('KAFKA_TOPIC', 'topic-name')

    consumer = KafkaConsumer(bootstrap_servers, group_id, topic)
    try:
        consumer.connect()
        consumer.consume()
    except Exception as e:
        logger.error(f"Failed to run consumer: {e}")
        consumer.close()
        raise

if __name__ == "__main__":
    main()