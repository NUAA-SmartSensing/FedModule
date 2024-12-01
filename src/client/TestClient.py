from client.NormalClient import NormalClient
from client.mixin.ClientHandler import UpdateReceiver, DelaySimulator, UpdateSender
from client.mixin.InitHandler import InitHandlerWithTest
from core.handlers.Handler import HandlerChain
from core.handlers.ModelTestHandler import ClientTestHandler, ClientPostTestHandler
from core.handlers.ModelTrainHandler import ClientTrainHandler, ClientPostTrainHandler


class TestClient(NormalClient):
    def create_handler_chain(self):
        self.init_chain = InitHandlerWithTest()
        self.handler_chain = HandlerChain()
        (self.handler_chain.set_chain(UpdateReceiver())
         .set_next(ClientTrainHandler())
         .set_next(ClientPostTrainHandler())
         .set_next(ClientTestHandler())
         .set_next(ClientPostTestHandler())
         .set_next(DelaySimulator())
         .set_next(UpdateSender()))
