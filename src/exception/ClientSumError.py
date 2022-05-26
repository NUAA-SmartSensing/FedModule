class ClientSumError(Exception):
    def __init__(self):
        super().__init__("the sum of stale->list is not equal to the client_num", 1)
        self.message = "the sum of stale->list is not equal to the client_num"
        self.status = 1
