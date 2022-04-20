from functools import wraps


class LogIncomingMessage(object):
    def __init__(self, logger, debug):
        self.logger = logger
        self.debug = debug

    def __call__(self, on_message_function):
        @wraps(on_message_function)
        def decorated(client, user_data, message, parsed_message):

            if self.debug:
                self.logger.info(parsed_message)

            return on_message_function(client, user_data, message, parsed_message)

        return decorated
