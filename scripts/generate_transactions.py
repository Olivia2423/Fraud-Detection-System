import json, time
from kafka import KafkaProducer
from fraud.data import synth_transactions
from fraud.config import KAFKA_BOOTSTRAP, KAFKA_TOPIC_TX

def main():
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8"),
        linger_ms=20,
        acks="all"
    )
    while True:
        df = synth_transactions(n=1, with_labels=False)
        rec = df.to_dict(orient="records")[0]
        producer.send(KAFKA_TOPIC_TX, key=rec["transaction_id"], value=rec)
        producer.flush()
        time.sleep(0.05)

if __name__ == "__main__":
    main()
