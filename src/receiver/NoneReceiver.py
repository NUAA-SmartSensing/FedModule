from receiver.AbstractReceiver import AbstractReceiver


class NoneReceiver(AbstractReceiver):
    def __init__(self, config):
        super().__init__(config)

    def receive(self):
        pass
