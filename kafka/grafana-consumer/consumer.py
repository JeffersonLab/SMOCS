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

# Set up Kafka consumer
# consumer = KafkaConsumer(
#     KAFKA_TOPIC,
#     bootstrap_servers=KAFKA_SERVERS,
#     auto_offset_reset='earliest',
#     group_id='grafana-consumer',
#     value_deserializer=lambda m: json.loads(m.decode('utf-8'))
# )

# print(f"Listening to Kafka topic '{KAFKA_TOPIC}'...")
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
# consumer = KafkaConsumer(
#     bootstrap_servers=KAFKA_SERVERS,
#     auto_offset_reset='earliest',
#     group_id='grafana-consumer',
#     value_deserializer=lambda m: json.loads(m.decode('utf-8') if m else None)
# )

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
    
    print(f"type of msg here: {type(value)}")

    if 'timeStamp' not in value or 'watchdog' not in value:
        print("Skipping message due to missing 'timeStamp' or 'watchdog' key.")
        continue
    
    timestamp = value['timeStamp']
    value = value['watchdog']['value']
    measurement = "counter"

    # Sanitize non-numeric values if needed
    influx_data = {
        "measurement": measurement,
        "tags": {"source": "kafka"},
        "fields": {}
    }
    influx_data["fields"]["timestamp"] = timestamp
    influx_data["fields"]["value"] = value

    # for k, v in value.items():
    #     if isinstance(v, (int, float, str)):
    #         influx_data["fields"][k] = v

    influx.write_points([influx_data])