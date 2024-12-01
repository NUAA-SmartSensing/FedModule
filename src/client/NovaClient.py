import copy

import torch

from client.NormalClient import NormalClient


class NovaClient(NormalClient):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        NormalClient.__init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.tau = 0

    def train(self):
        self.tau = 0
        global_model = copy.deepcopy(self.model.state_dict())
        data_sum = len(self.train_dl)
        for epoch in range(self.epoch):
            for data, label in self.train_dl:
                self.tau += 1
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.model(data)
                # Calculate the loss function
                loss = self.loss_func(preds, label)
                # proximal term
                if self.mu != 0:
                    proximal_term = 0.0
                    for w, w_t in zip(self.model.parameters(), global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)
                    loss = loss + (self.mu / 2) * proximal_term
                # backpropagate
                loss.backward()
                # Update the gradient
                self.optimizer.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                # Zero out the gradient and initialize the gradient.
                self.optimizer.zero_grad()
        # return the delta weights
        weights = copy.deepcopy(self.model.state_dict())
        for k, v in weights.items():
            weights[k] = weights[k] - global_model[k]
        return data_sum, weights

    def customize_upload(self):
        self.upload_item("tau", self.tau)


class NovaClientWithGrad(NovaClient):
    def train(self):
        self.tau = 0
        if self.mu != 0:
            global_model = copy.deepcopy(self.model)
        data_sum = len(self.train_dl)
        accumulated_grads = []  # Initialize the list of accumulated gradients

        # Traverse the training data.
        for data, label in self.train_dl:
            self.tau += 1
            data, label = data.to(self.dev), label.to(self.dev)
            preds = self.model(data)
            # Calculate the loss function
            loss = self.loss_func(preds, label)
            # Proximal term
            if self.mu != 0:
                proximal_term = 0.0
                for w, w_t in zip(self.model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)
                loss = loss + (self.mu / 2) * proximal_term
            # Backpropagate, but do not execute optimization steps.
            loss.backward()
            # Accumulate gradients.
            accumulated_grads = [None if acc_grad is None else acc_grad + param.grad
                                 for acc_grad, param in zip(accumulated_grads, self.model.parameters())]

            # Zero out the gradient to prepare for the next iteration.
            self.model.zero_grad()
        # return accumulate gradients.
        torch.cuda.empty_cache()
        return data_sum, accumulated_grads
