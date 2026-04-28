from kafka import KafkaProducer
import json
#collecting streaming data
producer = KafkaProducer(
  bootstrap_servers='localhost:9092",
  value_serializer=lambda v: json.dumps(v).encode('utf-8')
  )

  data = {"text": "you are stupid", "label":1}

procucer.send('cyberbullying-topic', data)
producer.flush()
