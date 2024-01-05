from abc import abstractmethod


class AbstractUpdate:
    @abstractmethod
    def update_server_weights(self, *args, **kwargs):
        """
        Implements abstract_method in subclass.
        Must return global_model, delivery_weights.
        global_model: The weights of the global model for the next epoch.
        delivery_weights: The weights to be delivered to the clients.
        """
        pass
