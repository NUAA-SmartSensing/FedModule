import wandb

from updater.SyncUpdater import SyncUpdater


class PersonalUpdater(SyncUpdater):
    def __init__(self, server_thread_lock, stop_event, config, mutex_sem, empty_sem, full_sem):
        SyncUpdater.__init__(self, server_thread_lock, stop_event, config, mutex_sem, empty_sem, full_sem)

    def server_update(self, epoch, update_list):
        self.update_server_weights(epoch, update_list)
        acc, loss = self.run_server_test(epoch)
        avg_acc, avg_loss = self.run_personalization_test(epoch, update_list)
        if self.config['enabled']:
            wandb.log({'accuracy': acc, 'loss': loss, 'avg_acc': avg_acc, 'avg_loss': avg_loss})

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
        return accuracy, loss

