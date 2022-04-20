class Configuration:
    broker_ip_address = ""
    broker_port = 1883
    broker_subscribe_topics = []
    broker_publish_topic = ""

    def __init__(
        self,
        broker_ip_address,
        broker_publish_topic,
        broker_subscribe_topics=[],
        broker_port=1883,
    ):
        self.broker_ip_address = broker_ip_address
        self.broker_publish_topic = broker_publish_topic
        self.broker_subscribe_topics = broker_subscribe_topics
        self.broker_port = broker_port