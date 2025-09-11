import json, time, requests
from kafka import KafkaConsumer, KafkaProducer
from .config import KAFKA_BOOTSTRAP, KAFKA_TOPIC_TX, KAFKA_TOPIC_SCORED, API_URL

def start_stream():
    consumer = KafkaConsumer(
        KAFKA_TOPIC_TX,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id="fraud-consumer-1",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
        auto_offset_reset="latest",
        enable_auto_commit=True,
    )
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    for msg in consumer:
        tx = msg.value
        try:
            r = requests.post(API_URL, json=tx, timeout=1.5)
            r.raise_for_status()
            scored = r.json()
            scored["ts"] = time.time()
            producer.send(KAFKA_TOPIC_SCORED, scored)
            if scored["fraud_probability"] >= 0.8:
                print("[ALERT] High-risk txn:", scored)
        except Exception as e:
            print("scoring failed:", e)
