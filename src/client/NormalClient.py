from client.Client import Client
from client.mixin.ClientHandler import UpdateReceiver, DelaySimulator, UpdateSender
from client.mixin.InitHandler import InitHandler
from core.handlers.Handler import HandlerChain
from core.handlers.ModelTrainHandler import ClientTrainHandler, ClientPostTrainHandler
from client.mixin.DataStore import DataStore


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

    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev, data_proxy=None):
        super().__init__(c_id, stop_event, selected_event, delay, index_list, dev)
        self.init_chain = HandlerChain()
        self.update_dict = {}
        self.lr_scheduler = None
        self.batch_size = config.get("batch_size", 64)
        self.epoch = config["epochs"]
        self.optimizer_config = config["optimizer"]
        self.mu = config.get("mu", 0)
        self.config = config
        self.data_proxy = data_proxy if data_proxy is not None else DataStore()

    @property
    def fl_train_ds(self):
        return self.data_proxy.get(self.client_id, 'fl_train_ds')

    @fl_train_ds.setter
    def fl_train_ds(self, value):
        self.data_proxy.set(self.client_id, 'fl_train_ds', value)

    @property
    def optimizer(self):
        return self.data_proxy.get(self.client_id, 'optimizer')

    @optimizer.setter
    def optimizer(self, value):
        self.data_proxy.set(self.client_id, 'optimizer', value)

    @property
    def loss_func(self):
        return self.data_proxy.get(self.client_id, 'loss_func')

    @loss_func.setter
    def loss_func(self, value):
        self.data_proxy.set(self.client_id, 'loss_func', value)

    @property
    def train_dl(self):
        return self.data_proxy.get(self.client_id, 'train_dl')

    @train_dl.setter
    def train_dl(self, value):
        self.data_proxy.set(self.client_id, 'train_dl', value)

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

    def run_one_iteration(self, client_dict=None):
        """
        通过调用方式运行一次迭代，并与外部共享数据以减少内存使用。

        参数:
            global_var: 可选，外部传入的全局变量（模型参数等）
            time_stamp: 可选，外部传入的时间戳

        返回:
            update_dict: 客户端更新的结果字典
        """
        # 直接使用传入的外部数据引用而不是复制
        if client_dict is not None:
            for k, v in client_dict.items():
                setattr(self, k, v)

        # 执行训练流程
        self.message_queue.set_training_status(self.client_id, True)
        self.execute_chain()
        self.message_queue.set_training_status(self.client_id, False)

        # 返回更新字典供外部使用
        return self


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
