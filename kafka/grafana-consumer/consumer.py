import os
from kafka import KafkaConsumer
from influxdb import InfluxDBClient
import json
import time
import ast

# Environment variables
KAFKA_TOPIC = os.environ.get('KAFKA_TOPIC')  # Not used in this example, but can be set if needed
KAFKA_SERVERS = os.environ.get('KAFKA_BROKER_URL', 'kafka-broker:9092')
INFLUXDB_DB = os.environ.get('INFLUXDB_DB', 'kafka_data')
INFLUXDB_USER = os.environ.get('INFLUXDB_USER')
INFLUXDB_PASS = os.environ.get('INFLUXDB_PASS')

# Set up InfluxDB client
influx = InfluxDBClient(host='influxdb', port=8086, username=INFLUXDB_USER, password=INFLUXDB_PASS)
influx.switch_database(INFLUXDB_DB)

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
    value = msg.value #.decode("utf-8")
    topic = msg.topic
    print("Received message:", topic, type(value), value)  # Show escaped string
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
    rate1 = value['rate1']['value']
    rate1_low = value['rate1']['alarm']['limits']['low']
    rate1_high = value['rate1']['alarm']['limits']['high']
    rate2 = value['rate2']['value']
    rate2_low = value['rate2']['alarm']['limits']['low']
    rate2_high = value['rate2']['alarm']['limits']['high']
    rate3 = value['rate3']['value']
    rate3_low = value['rate3']['alarm']['limits']['low']
    rate3_high = value['rate3']['alarm']['limits']['high']
    
    measurement = "FlowRates"

    # Sanitize non-numeric values if needed
    influx_data = {
        "measurement": measurement,
        "tags": {
            "topic": topic
        },
        "fields": {
            "rate1_value": rate1,
            "rate2_value": rate2,
            "rate3_value": rate3,
            "rate1_low": rate1_low,
            "rate1_high": rate1_high,
            "rate2_low": rate2_low,
            "rate2_high": rate2_high,
            "rate3_low": rate3_low,
            "rate3_high": rate3_high
        },
        # "time": timestamp
    }

    
    # influx_data["fields"]["timestamp"] = timestamp
    # influx_data["fields"]["value"] = value

    # for k, v in value.items():
    #     if isinstance(v, (int, float, str)):
    #         influx_data["fields"][k] = v

    influx.write_points([influx_data])