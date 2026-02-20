import os
import logging
import json
import argparse
from datetime import datetime
from typing import List, Dict

from jlab_archiver_client import MySampler, MySamplerQuery

from smocs.cores import KafkaProducerBase
from smocs.utils import setup_logging


class MyaKafkaProducer(KafkaProducerBase):
    """
    MYA Archiver to Kafka producer for historical EPICS data.

    Queries JLab MYA archiver for historical PV data and publishes to Kafka.
    CLI-driven one-shot execution.

    This producer:
    - Executes a single MYA archiver query based on CLI arguments
    - Converts each timestamp in the result to a separate Kafka message
    - Publishes all messages to a Kafka topic
    - Exits after completion
    """

    def __init__(self, args):
        """
        Initialize the MYA Kafka producer.

        Args:
            args: Parsed argparse namespace with CLI arguments
        """
        # Initialize base class with Kafka broker URL
        super().__init__(args.kafka_broker)

        # Store CLI arguments
        self.start_time = args.start_time
        self.interval = args.interval
        self.num_samples = args.num_samples
        self.pvlist = args.pvlist
        self.kafka_topic = args.kafka_topic

        logging.info(f"MYA Producer initialized:")
        logging.info(f"  Start time: {self.start_time}")
        logging.info(f"  Interval: {self.interval} ms")
        logging.info(f"  Number of samples: {self.num_samples}")
        logging.info(f"  PV list: {self.pvlist}")
        logging.info(f"  Kafka topic: {self.kafka_topic}")

    def build_mya_query(self) -> MySamplerQuery:
        """
        Build MySamplerQuery from CLI arguments.

        Returns:
            MySamplerQuery: Configured query object

        Raises:
            ValueError: If start_time format is invalid
        """
        try:
            # Parse start time string to datetime object
            start_dt = datetime.strptime(self.start_time, "%Y-%m-%d %H:%M:%S")

            # Build query
            query = MySamplerQuery(
                start=start_dt,
                interval=self.interval,
                num_samples=self.num_samples,
                pvlist=self.pvlist
            )

            logging.info(f"Built MYA query for {len(self.pvlist)} PVs")
            return query

        except ValueError as e:
            logging.error(f"Invalid start_time format. Expected 'YYYY-MM-DD HH:MM:SS', got '{self.start_time}'")
            raise ValueError(f"Invalid datetime format: {e}")

    def execute_mya_query(self, query: MySamplerQuery):
        """
        Execute MYA query and return results.

        Args:
            query: MySamplerQuery object

        Returns:
            pd.DataFrame: Query results with datetime index and PV columns

        Raises:
            Exception: If MYA query fails
        """
        try:
            logging.info(f"Executing MYA query for {len(self.pvlist)} PVs from {self.start_time}")

            # Create MySampler and execute query
            mysampler = MySampler(query)
            mysampler.run()

            # Get results
            df = mysampler.data

            # Log metadata
            logging.info(f"Retrieved {len(df)} samples from MYA archiver")
            logging.debug(f"MYA metadata: {mysampler.metadata}")

            # Check for disconnects
            if mysampler.disconnects:
                logging.warning(f"MYA disconnects detected: {mysampler.disconnects}")

            return df

        except Exception as e:
            logging.error(f"MYA query failed: {e}")
            raise

    def dataframe_to_messages(self, df) -> List[Dict]:
        """
        Convert MYA DataFrame to list of Kafka messages.

        Each row (timestamp) in the DataFrame becomes one message with all PV values.

        Args:
            df: pandas DataFrame with datetime index and PV columns

        Returns:
            List[Dict]: List of message dictionaries
        """
        messages = []

        for idx, row in df.iterrows():
            # Convert datetime index to Unix timestamp
            if hasattr(idx, 'timestamp'):
                timestamp = idx.timestamp()
            else:
                # Fallback for numeric index
                timestamp = float(idx)

            # Build channels dict from row values
            channels = row.to_dict()

            # Build message in standard SMOCS format
            message = {
                'timestamp': timestamp,
                'channels': channels,
                'source_topic': self.kafka_topic
            }

            messages.append(message)

        logging.debug(f"Converted DataFrame to {len(messages)} messages")
        if messages:
            logging.debug(f"Sample message: {messages[0]}")

        return messages

    def send_messages_to_kafka(self, messages: List[Dict], topic_name: str):
        """
        Send list of messages to Kafka topic.

        Args:
            messages: List of message dictionaries
            topic_name: Kafka topic name
        """
        if not messages:
            logging.warning("No messages to send to Kafka")
            return

        # Sanitize topic name
        sanitized_topic = self.sanitize_topic_name(topic_name)

        # Ensure topic exists
        self.create_topic_if_not_exists(sanitized_topic)

        logging.info(f"Sending {len(messages)} messages to Kafka topic '{sanitized_topic}'")

        # Send each message
        sent_count = 0
        failed_count = 0

        for i, message in enumerate(messages):
            try:
                # JSON serialize message
                message_json = json.dumps(message)

                # Send to Kafka
                record_metadata = self.send_to_kafka(sanitized_topic, message_json)

                sent_count += 1

                if i == 0:
                    # Log first message details
                    logging.info(f"First message sent to partition {record_metadata.partition}, offset {record_metadata.offset}")

            except Exception as e:
                failed_count += 1
                logging.error(f"Failed to send message {i}: {e}")

        # Summary
        logging.info(f"Kafka send complete: {sent_count} sent, {failed_count} failed")

    def start(self):
        """
        Start the MYA to Kafka producer (one-shot execution).

        This method:
        1. Sets up Kafka producer
        2. Builds and executes MYA query
        3. Converts results to messages
        4. Sends messages to Kafka
        5. Cleans up and exits
        """
        try:
            logging.info("Starting MYA-to-Kafka producer")

            # Setup Kafka producer using base class method
            logging.info("Setting up Kafka producer...")
            self.setup_kafka_producer()

            # Build MYA query
            query = self.build_mya_query()

            # Execute query
            df = self.execute_mya_query(query)

            # Check if DataFrame is empty
            if df.empty:
                logging.warning("MYA query returned no data. No messages to send.")
                return

            # Convert DataFrame to messages
            messages = self.dataframe_to_messages(df)

            # Send messages to Kafka
            self.send_messages_to_kafka(messages, self.kafka_topic)

            logging.info("MYA-to-Kafka producer completed successfully")

        except KeyboardInterrupt:
            logging.info("Interrupted by user")
            self.cleanup()
        except Exception as e:
            logging.error(f"Error in MYA producer: {e}", exc_info=True)
            self.cleanup()
            raise


def main():
    """
    Main entry point for the MYA to Kafka producer.
    """
    setup_logging()

    # Parse CLI arguments
    parser = argparse.ArgumentParser(
        description='MYA Archiver to Kafka Producer - Query historical EPICS data and publish to Kafka',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic query
  python -m smocs.producers.mya_kafka_producer \\
    --start-time "2019-08-12 00:00:00" \\
    --interval 1800000 \\
    --num-samples 15 \\
    --pvlist R12XGMES R13XGMES

  # Custom topic and broker
  python -m smocs.producers.mya_kafka_producer \\
    --start-time "2019-08-12 00:00:00" \\
    --interval 1800000 \\
    --num-samples 15 \\
    --pvlist IPM2C24A.XPOS IPM2C24A.YPOS \\
    --kafka-topic beam-position \\
    --kafka-broker localhost:9092
        """
    )

    parser.add_argument(
        '--start-time',
        required=True,
        help='Query start time (format: "YYYY-MM-DD HH:MM:SS")'
    )

    parser.add_argument(
        '--interval',
        type=int,
        required=True,
        help='Interval between samples in milliseconds'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        required=True,
        help='Number of samples to retrieve'
    )

    parser.add_argument(
        '--pvlist',
        nargs='+',
        required=True,
        help='Space-separated list of PV names to query'
    )

    parser.add_argument(
        '--kafka-topic',
        default='mya-archival',
        help='Kafka topic name (default: mya-archival)'
    )

    parser.add_argument(
        '--kafka-broker',
        default=None,
        help='Kafka broker URL (default: from KAFKA_BROKER_URL env var or kafka-broker:9092)'
    )

    args = parser.parse_args()

    # Use environment variable if kafka-broker not provided
    if args.kafka_broker is None:
        args.kafka_broker = os.getenv('KAFKA_BROKER_URL', 'kafka-broker:9092')

    logging.info("Starting MYA Archiver-to-Kafka producer")

    # Create and start producer
    producer = MyaKafkaProducer(args)
    producer.start()


if __name__ == "__main__":
    main()
