from threading import Thread


class MPMT(Thread):
    def __init__(self, client):
        super().__init__()
        self.client = client

    def run(self):
        self.client.run()
