from client.Client import Client
from client.mixin.ClientHandler import UpdateReceiver, DelaySimulator, UpdateSender
from client.mixin.InitHandler import InitHandler
from core.handlers.Handler import HandlerChain
from core.handlers.ModelTrainHandler import ClientTrainHandler, ClientPostTrainHandler


class NormalClient(Client):
    r"""
    This class inherits from Client and implements a basic Client.

    Attributes:
    fl_train_ds: FLDataset
        Local training dataset
    opti: Object
        Optimizer
    loss_func: Object
        Loss function
    train_dl: torch.utils.data.DataLoader
        Training data loader
    batch_size: int
        Batch size
    epoch: int
        Epoch number
    optimizer_config: dict
        Optimizer configuration
    mu: float
        Regularization coefficient
    config: dict
        Configuration
    dev: str
        Device
    """

    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        super().__init__(c_id, stop_event, selected_event, delay, index_list, dev)
        self.init_chain = HandlerChain()
        self.update_dict = {}
        self.lr_scheduler = None
        self.fl_train_ds = None
        self.optimizer = None
        self.loss_func = None
        self.train_dl = None
        self.batch_size = config.get("batch_size", 64)
        self.epoch = config["epochs"]
        self.optimizer_config = config["optimizer"]
        self.mu = config.get("mu", 0)
        self.config = config

    def _run_iteration(self):
        while not self.stop_event.is_set():
            if self.event.is_set():
                self.event.clear()
                self.local_run()
            else:
                self.event.wait()

    def local_run(self):
        """
        The run function of Client runs the main body, suitable for use as a target parameter of process.
        """
        self.message_queue.set_training_status(self.client_id, True)
        self.execute_chain()
        self.message_queue.set_training_status(self.client_id, False)

    def execute_chain(self):
        request = {"global_var": self.global_var, "client": self, 'epoch': self.time_stamp}
        self.handler_chain.handle(request)

    def init(self):
        request = {"global_var": self.global_var, "client": self, "config": self.config}
        self.init_chain.handle(request)

    def receive_notify(self):
        pass

    def create_handler_chain(self):
        self.init_chain = HandlerChain(InitHandler())
        self.handler_chain = HandlerChain()
        (self.handler_chain.set_chain(UpdateReceiver())
         .set_next(ClientTrainHandler())
         .set_next(ClientPostTrainHandler())
         .set_next(DelaySimulator())
         .set_next(UpdateSender()))

    def finish(self):
        pass

    def upload(self, **kwargs):
        """
        The detailed parameters uploaded to the server by Client.
        """
        self.update_dict["client_id"] = self.client_id
        self.update_dict["time_stamp"] = self.time_stamp
        for k, v in kwargs.items():
            self.upload_item(k, v)
        self.customize_upload()
        self.message_queue.put_into_uplink(self.update_dict)
        print("Client", self.client_id, "uploaded")

    def upload_item(self, k, v):
        self.update_dict[k] = v

    def customize_upload(self):
        """
        Customize the parameters uploaded to the server by the client.
        """
        pass


class NormalClientWithDelta(NormalClient):
    """
    This class inherits from NormalClient and implements the functionality of uploading
    the difference between model parameters before and after training, i.e.,
    uploads $delta = w_i^t - w^t$, where $w_i^t$ is the model parameter updated by the client locally,
    and $w^t$ is the model parameter before local iteration.
    """
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        super().__init__(c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.config['train_func'] = 'core.handlers.ModelTrainHandler.TrainWithDelta'


class NormalClientWithGrad(NormalClient):
    """
    This class inherits from NormalClient and implements the cumulative gradient upload after training.
    """
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        super().__init__(c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.config['train_func'] = 'core.handlers.ModelTrainHandler.TrainWithGrad'
