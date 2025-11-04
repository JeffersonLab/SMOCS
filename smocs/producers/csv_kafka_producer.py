import os
import logging
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from smocs.cores import KafkaProducerBase
from smocs.utils import setup_logging


class CSVKafkaProducer(KafkaProducerBase):
    """
    CSV to Kafka producer that reads time series data from CSV files.
    
    This producer reads CSV files containing time series data, filters by date range,
    and sends each row as a structured message to Kafka topics.
    """
    
    def __init__(self):
        """
        Initialize the CSV to Kafka producer with configuration.
        """
        # Initialize base class
        kafka_broker_url = os.getenv('KAFKA_BROKER_URL', 'kafka-broker:9092')
        super().__init__(kafka_broker_url)
        
        # CSV file path - prioritize environment variable
        self.csv_file_path = os.getenv('CSV_FILE_PATH', '/app/data/2024-05-14-rf-ndx-raw-supplement.csv')
        
        # Date configuration
        self.date_column = os.getenv('CSV_DATE_COLUMN', 'Date')
        self.date_format = os.getenv('CSV_DATE_FORMAT', '%Y-%m-%d_%H:%M:%S')
        
        # Date range filtering (optional)
        self.start_date_str = os.getenv('START_DATE')
        self.end_date_str = os.getenv('END_DATE')
        
        self.start_date = None
        self.end_date = None
        
        if self.start_date_str:
            self.start_date = datetime.strptime(self.start_date_str, self.date_format)
            logging.info(f"Filtering from start date: {self.start_date}")
        
        if self.end_date_str:
            self.end_date = datetime.strptime(self.end_date_str, self.date_format)
            logging.info(f"Filtering to end date: {self.end_date}")
        
        # Kafka topic configuration
        self.kafka_topic = os.getenv('CSV_KAFKA_TOPIC', 'csv_timeseries_data')
        self.source_name = os.getenv('CSV_SOURCE_NAME', 'csv_file_producer')
        
        # Columns to exclude (comma-separated list in env var)
        exclude_columns_str = os.getenv('CSV_EXCLUDE_COLUMNS', 'level_0')
        self.exclude_columns = [col.strip() for col in exclude_columns_str.split(',') if col.strip()]
        
        # Sending configuration
        self.send_delay = float(os.getenv('CSV_SEND_DELAY', '0.0'))
        self.loop_playback = os.getenv('CSV_LOOP_PLAYBACK', 'false').lower() == 'true'
        
        logging.info(f"CSV file: {self.csv_file_path}")
        logging.info(f"Date column: {self.date_column}")
        logging.info(f"Date format: {self.date_format}")
        logging.info(f"Kafka topic: {self.kafka_topic}")
        logging.info(f"Source name: {self.source_name}")
        logging.info(f"Excluded columns: {self.exclude_columns}")
        logging.info(f"Send delay: {self.send_delay}s")
        logging.info(f"Loop playback: {self.loop_playback}")
    
    def parse_csv_row(self, row: Dict[str, str], headers: List[str]) -> Optional[Dict[str, Any]]:
        """
        Parse a CSV row into a structured message for Kafka.
        
        Args:
            row: Dictionary representing a CSV row
            headers: List of column headers
            
        Returns:
            Structured message dict or None if row should be filtered
        """
        try:
            # Extract and parse timestamp
            date_str = row[self.date_column]
            timestamp = datetime.strptime(date_str, self.date_format)
            
            # Filter by date range if specified
            if self.start_date and timestamp < self.start_date:
                return None
            if self.end_date and timestamp > self.end_date:
                return None
            
            # Build channels dictionary (all columns except date and excluded ones)
            channels = {}
            for column in headers:
                if column == self.date_column or column in self.exclude_columns:
                    continue
                
                value = row[column]
                
                # Try to convert to numeric type
                try:
                    # Try integer first
                    if '.' not in value:
                        channels[column] = int(value)
                    else:
                        channels[column] = float(value)
                except ValueError:
                    # Keep as string if not numeric
                    channels[column] = value
            
            # Build output message
            message = {
                "timestamp": timestamp.isoformat(),
                "channels": channels,
                "source_topic": self.source_name
            }
            
            return message
            
        except KeyError as e:
            logging.error(f"Missing required column in CSV row: {e}")
            return None
        except ValueError as e:
            logging.error(f"Error parsing date in CSV row: {e}")
            return None
        except Exception as e:
            logging.error(f"Error parsing CSV row: {e}")
            return None
    
    def process_csv_file(self):
        """
        Read and process the CSV file, sending each row to Kafka in order.
        
        Returns:
            Tuple of (total_rows, sent_rows, filtered_rows, error_rows)
        """
        csv_path = Path(self.csv_file_path)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")
        
        total_rows = 0
        sent_rows = 0
        filtered_rows = 0
        error_rows = 0
        
        logging.info(f"Processing CSV file: {self.csv_file_path}")
        
        # Sanitize Kafka topic name
        kafka_topic = self.sanitize_topic_name(self.kafka_topic)
        
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            headers = reader.fieldnames
            
            logging.info(f"CSV columns: {headers}")
            
            for row in reader:
                total_rows += 1
                
                # Parse the row
                parsed_message = self.parse_csv_row(row, headers)
                
                if parsed_message is None:
                    filtered_rows += 1
                    continue
                
                try:
                    # Convert to JSON
                    kafka_message = json.dumps(parsed_message)
                    
                    # Send to Kafka
                    record_metadata = self.send_to_kafka(kafka_topic, kafka_message)
                    
                    sent_rows += 1
                    
                    if sent_rows % 100 == 0:
                        logging.info(f"Progress: {sent_rows} rows sent to Kafka")
                    
                    logging.debug(
                        f'Sent to Kafka topic "{kafka_topic}" - '
                        f'timestamp {parsed_message["timestamp"]} - '
                        f'partition {record_metadata.partition}, '
                        f'offset {record_metadata.offset}'
                    )
                    
                    # Add delay if configured
                    if self.send_delay > 0:
                        import time
                        time.sleep(self.send_delay)
                    
                except Exception as e:
                    error_rows += 1
                    logging.error(f"Error sending row {total_rows} to Kafka: {e}")
                    logging.error(f"Row data: {row}")
        
        return total_rows, sent_rows, filtered_rows, error_rows
    
    def start(self):
        """
        Start the CSV to Kafka producer.
        
        This method:
        1. Sets up Kafka producer (via base class)
        2. Reads the CSV file
        3. Processes and sends each row to Kafka in order
        4. Optionally loops if configured
        """
        try:
            logging.info("Starting CSV-to-Kafka producer")
            
            # Setup Kafka producer using base class method
            logging.info("Setting up Kafka producer...")
            self.setup_kafka_producer()
            
            # Process loop
            iteration = 0
            while True:
                iteration += 1
                if self.loop_playback and iteration > 1:
                    logging.info(f"Starting playback loop iteration {iteration}")
                
                # Process the CSV file
                logging.info("Processing CSV file...")
                total, sent, filtered, errors = self.process_csv_file()
                
                # Log summary
                logging.info("=" * 60)
                logging.info(f"CSV Processing Summary (Iteration {iteration}):")
                logging.info(f"  Total rows read:     {total}")
                logging.info(f"  Rows sent to Kafka:  {sent}")
                logging.info(f"  Rows filtered:       {filtered}")
                logging.info(f"  Rows with errors:    {errors}")
                logging.info("=" * 60)
                
                if errors > 0:
                    logging.warning(f"{errors} rows failed to process. Check logs for details.")
                
                # Break if not looping
                if not self.loop_playback:
                    logging.info("CSV processing complete. Shutting down...")
                    break
                
                logging.info("Restarting playback from beginning...")
            
            self.cleanup()
            
        except KeyboardInterrupt:
            logging.info("Interrupted. Shutting down...")
            self.cleanup()
        except Exception as e:
            logging.error(f"Error: {e}")
            self.cleanup()
            raise


def main():
    """
    Main entry point for the CSV to Kafka producer.
    """
    setup_logging()
    logging.info("Starting CSV-to-Kafka producer")
    
    producer = CSVKafkaProducer()
    producer.start()


if __name__ == "__main__":
    main()