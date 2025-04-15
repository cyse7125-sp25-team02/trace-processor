# trace-processor/consumer.py
import json
import logging
import os
import time
import tempfile
from confluent_kafka import Consumer, KafkaException
from typing import Dict
from pdf_to_text import process_pdf_pipeline  # Import the pipeline function
from google.cloud import storage  # For GCS download

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
            'auto.offset.reset': 'latest',
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
    
    def download_from_gcs(self, gcs_url: str, local_path: str) -> bool:
        """Download a file from a GCS URL to a local path."""
        try:
            # Convert https://storage.googleapis.com/bucket_name/blob to bucket_name and blob_name
            if not gcs_url.startswith("https://storage.googleapis.com/"):
                logger.error(f"Invalid GCS URL: {gcs_url}")
                return False
            path = gcs_url.replace("https://storage.googleapis.com/", "")
            bucket_name, blob_name = path.split("/", 1)
            
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded {gcs_url} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download {gcs_url}: {e}", exc_info=True)
            return False

    def process_message(self, message: Dict) -> None:
        """
        Process a Kafka message containing GCS bucket path and metadata, download the PDF,
        and run the PDF processing pipeline.
        """
        logger.info(f"Processing JSON message: {message}")

        try:
            # Validate metadata fields (optional, but ensures compatibility)
            required_metadata = [
                'instructor_name', 'course_code', 'semester_term', 
                'semester_year', 'course_name', 'credit_hours', 'bucket_path'
            ]
            
            for key in required_metadata:
                if key not in message:
                    logger.warning(f"Metadata missing '{key}' field. Proceeding anyway.")

            bucket_path = message.get('bucket_path')
            if not bucket_path:
                logger.error("Message missing 'bucket_path' field.")
                return
            
            # Download the PDF from GCS to a temporary local file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                local_pdf_path = temp_file.name
            
            try:
                if not self.download_from_gcs(bucket_path, local_pdf_path):
                    logger.info("local_pdf_path: %s", local_pdf_path)
                    logger.error(f"Failed to download PDF from {bucket_path}")
                    return

                logger.info(f"Processing PDF: {bucket_path} with metadata: {message}")
                
                # Call the PDF processing pipeline with the local file path
                chunks_with_embeddings = process_pdf_pipeline(local_pdf_path, message)
                
                if chunks_with_embeddings:
                    logger.info(f"Successfully processed PDF. Generated {len(chunks_with_embeddings)} chunks.")
                    # Optionally, print the first few chunks for verification
                    for i, chunk in enumerate(chunks_with_embeddings[:5]):
                        logger.info(f"--- Chunk {i+1} (with Embedding) ---")
                        logger.info(f"ID: {chunk['id']}")
                        logger.info(f"Text: {chunk['text']}")
                        embedding_preview = chunk.get('embedding')
                        if embedding_preview:
                            logger.info(f"Embedding (first 5 dims): {embedding_preview[:5]}...")
                            logger.info(f"Embedding dimensions: {len(embedding_preview)}")
                        else:
                            logger.info("Embedding: Error or not generated.")
                        logger.info(f"Metadata: {chunk['metadata']}")
                        logger.info("-" * 20)
                else:
                    logger.error("Failed to process PDF or generate chunks.")
            
            finally:
                # Clean up the temporary file
                try:
                    if os.path.exists(local_pdf_path):
                        os.unlink(local_pdf_path)
                        logger.info(f"Deleted temporary file: {local_pdf_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {local_pdf_path}: {e}")

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)

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