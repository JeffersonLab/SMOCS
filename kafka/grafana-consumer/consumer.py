import os
from kafka import KafkaConsumer
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import json
import time
import ast

# Environment variables
KAFKA_TOPIC = os.environ.get('KAFKA_TOPIC')
KAFKA_SERVERS = os.environ.get('KAFKA_BROKER_URL', 'kafka-broker:9092')
INFLUXDB_URL = os.environ.get('INFLUXDB_URL', 'http://influxdb:8086')
INFLUXDB_TOKEN = os.environ.get('INFLUXDB_TOKEN')
INFLUXDB_ORG = os.environ.get('INFLUXDB_ORG', 'myorg')
INFLUXDB_BUCKET = os.environ.get('INFLUXDB_BUCKET', 'kafka_data')

# Set up InfluxDB client
client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

connect = False
counter = 0
while not connect:
    try:
        consumer = KafkaConsumer(
            bootstrap_servers=KAFKA_SERVERS,
            auto_offset_reset='earliest',
            group_id='grafana-consumer',
            value_deserializer=lambda m: m.decode('utf-8') if m else None
        )
        connect = True
    except Exception as e:
        print(f"Waiting for Kafka broker: {e}. Retrying attempt {counter}...")
        counter += 1
        time.sleep(5)

consumer.subscribe(pattern=r'.*')
print("Subscribed to ALL Kafka topics...")

for msg in consumer:
    print(f"msg: {msg}")
    value = msg.value
    topic = msg.topic
    print("Received message:", topic, type(value), value)
    
    if value[:7] == "Message":
        print("Received 'Message 0', skipping...")
        continue
    
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    elif isinstance(value, str):
        value = ast.literal_eval(value)
    else:
        print("Received message of unexpected type:", type(value))
        continue

    if 'timeStamp' not in value or 'type' not in value:
        print("Skipping message due to missing 'timeStamp' or 'type' key.")
        continue
    elif value['type'] != 'blinky-mqtt':
        print(f"Skipping message with type '{value['type']}' (not 'blinky-mqtt').")
        continue
    
    timestamp = value['timeStamp']
    
    # Create point using the new Point API
    point = Point("FlowRates") \
        .tag("topic", topic) \
        .field("rate1_value", value['rate1']['value']) \
        .field("rate1_low", value['rate1']['alarm']['limits']['low']) \
        .field("rate1_high", value['rate1']['alarm']['limits']['high']) \
        .field("rate2_value", value['rate2']['value']) \
        .field("rate2_low", value['rate2']['alarm']['limits']['low']) \
        .field("rate2_high", value['rate2']['alarm']['limits']['high']) \
        .field("rate3_value", value['rate3']['value']) \
        .field("rate3_low", value['rate3']['alarm']['limits']['low']) \
        .field("rate3_high", value['rate3']['alarm']['limits']['high'])
    
    # If you want to use the original timestamp, add:
    # .time(timestamp)
    
    try:
        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
        print(f"Successfully wrote data to InfluxDB bucket: {INFLUXDB_BUCKET}")
    except Exception as e:
        print(f"Error writing to InfluxDB: {e}")