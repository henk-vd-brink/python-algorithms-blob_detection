import json
from functools import wraps


class ParseIncomingMessage(object):
    def __call__(self, on_message_function):
        @wraps(on_message_function)
        def decorated(client, user_data, message):
            incoming_message = message.payload.decode("utf-8")
            message_dict = incoming_message.replace("'", '"')
            parsed_message = json.loads(message_dict)
            return on_message_function(client, user_data, message, parsed_message)

        return decorated
