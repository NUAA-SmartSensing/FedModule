from client.NormalClient import NormalClient


class TWAFLClient(NormalClient):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        NormalClient.__init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.flag = False

    def local_task(self):
        self.flag = self.time_stamp % 15 in [11, 13, 14, 12, 0]
        # The client performs training.
        data_sum, weights = self.train()

        # Information transmitted from the client to the server has latency.
        print("Client", self.client_id, "trained")
        if self.flag:
            self.delay_simulate(self.delay)
        else:
            self.delay_simulate(self.delay / 2)

        # upload its updates
        self.upload(data_sum, weights)
