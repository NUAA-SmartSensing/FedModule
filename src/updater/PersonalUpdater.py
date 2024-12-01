import wandb

from core.handlers.Handler import Handler
from core.handlers.ModelTestHandler import ServerPostTestHandler
from updater.SyncUpdater import SyncUpdater
from utils.GlobalVarGetter import GlobalVarGetter


class PersonalUpdater(SyncUpdater):
    def create_handler_chain(self):
        super().create_handler_chain()
        self.handler_chain.add_handler_after(LocalTestCollector(), ServerPostTestHandler)


class LocalTestCollector(Handler):
    def __init__(self):
        super().__init__()
        global_var = GlobalVarGetter.get()
        self.cloud_enabled = global_var['config']['wandb']['enabled']
        self.file_enabled = global_var['config']['global']['save']
        self.accuracy_list = []
        self.loss_list = []

    def _handle(self, request):
        update_list = request.get('update_list')
        epoch = request.get('epoch')
        _ = self.run_personalization_test(epoch, update_list)
        return request

    def run_personalization_test(self, epoch, update_list):
        accuracy = 0
        loss = 0
        for i in range(len(update_list)):
            accuracy += update_list[i]["accuracy"]
            loss += update_list[i]["loss"]
        accuracy = accuracy / len(update_list)
        loss = loss / len(update_list)
        self.loss_list.append(loss)
        self.accuracy_list.append(accuracy)
        print('Epoch(t):', epoch, 'avg-accuracy:', accuracy, 'avg-loss', loss)
        if self.cloud_enabled:
            wandb.log({'avg-acc': accuracy, 'avg-loss': loss}, step=epoch)
        return accuracy, loss
