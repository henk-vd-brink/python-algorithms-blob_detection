import typing

import paho.mqtt.client as paho


class Client(paho.Client):  # type: typing.Any
    def __init__(
        self,
        config: typing.Any,
        logger: typing.Any = print,
        **kwargs: typing.Dict[str, typing.Any],
    ) -> None:
        super().__init__()

        self._config = config
        self.logger = logger

    def on_connect(
        self,
        client: typing.Any,
        userdata: typing.Any,
        flags: typing.Any,
        rc: int,
    ) -> None:
        """Overrides the default on_connect function"""
        self.logger(
            f"""[MQTT] Connecting to:
            - address: {self._config.broker_ip_address}:{self._config.broker_port}
            - subscribe topics: {self._config.broker_subscribe_topics}
            - publish topic: {self._config.broker_publish_topic}"""
        )

        if rc == 0:
            client.connected_flag = True
            for subscibe_topic in self._config.broker_subscribe_topics:
                client.subscribe((subscibe_topic, 0))
                self.logger(f"[MQTT] Succesfully connected to topic: {subscibe_topic}")

    def on_disconnect(
        self, client: typing.Any, userdata: typing.Any, rc: typing.Any
    ) -> None:
        """Overrides the default on_disconnect function"""
        client.connected_flag = False
        self.logger("[MQTT] Succesfully disconnected")

    def on_socket_close(
        self, client: typing.Any, userdata: typing.Any, sock: typing.Any
    ) -> None:
        """Overrides the default on_socket_close function"""
        self.logger("[MQTT] Socket closed")

    def on_socket_open(
        self, client: typing.Any, userdata: typing.Any, sock: typing.Any
    ) -> None:
        """Overrides the default on_socket_open function"""
        self.logger("[MQTT] Socket opened")

    def on_message(
        self, client: typing.Any, user_data: typing.Any, message: typing.Any
    ) -> None:
        """Overrides the default on_message function"""
        self._on_message_function(client, user_data, message)

    def bind_on_message(self, on_message_function_object: typing.Any) -> None:
        """Binds a new on_connect function to the class

        Args:
            on_message_function_object (typing.Any): on_message function
        """
        self._on_message_function = on_message_function_object

    def connect(self) -> None:
        super().connect(self._config.broker_ip_address, self._config.broker_port)