import copy
import gc
import pickle
import logging
import random
import sys
import os
import torch
import torch.nn as nn
import yaml
from torch import sign

from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Client(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID(对数据进行了打乱) compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """

    def __init__(self, client_id, local_data, device):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.data = local_data
        self.device = device
        self.__model = None

    @property
    def model(self):  # 模型获取
        """Local model getter for parameter aggregation."""
        return self.__model

    @model.setter
    def model(self, model):  # 模型设置
        """Local model setter for passing globally aggregated model parameters."""
        self.__model = model

    def __len__(self):  # 用户本地数据长度获取
        """Return a total size of the client's local data."""
        return len(self.data)

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        self.dataloader = DataLoader(self.data, batch_size=client_config["batch_size"], shuffle=True)
        self.local_epoch = client_config["num_local_epochs"]
        self.criterion = client_config["criterion"]
        self.optimizer = client_config["optimizer"]
        self.optim_config = client_config["optim_config"]

    def client_update(self):  # 本地训练
        """Update local model using local dataset."""
        self.model.train()  # 开始训练
        self.model.to(self.device)  # 放到GPU上
        with open('config.yaml') as c:
            configs = list(yaml.load_all(c, Loader=yaml.FullLoader))
        attack_method = configs[7]["attack_method"]
        optimizer = eval(self.optimizer)(self.model.parameters(),
                                         **self.optim_config)  # 第一个括号是eval(),返回一个优化器，第二个括号是优化器的参数

        if self.id in [i for i in range(0,3)]:  # 恶意用户是1,2
            if attack_method["attack"] == "Gaussian_attack":
                local_model_vector = torch.nn.utils.parameters_to_vector(self.model.parameters())
                local_model_vector = torch.randn(local_model_vector.shape[0])*(-0.1)
                torch.nn.utils.vector_to_parameters(local_model_vector, self.model.parameters())
            elif attack_method["attack"] == "Label_flipping":
                for e in range(self.local_epoch):
                    for data, labels in self.dataloader:
                        labels = torch.ones(len(labels))
                        r_label = random.randint(a=0, b=9)

                        labels = torch.ones(len(labels)) * r_label  # 标签反转攻击

                        data, labels = data.float().to(self.device), labels.long().to(self.device)

                        optimizer.zero_grad()
                        outputs = self.model(data)

                        loss = eval(self.criterion)()(outputs, labels)
                        loss.backward()  # 反向传播
                        optimizer.step()

                        if self.device == "cuda":
                            torch.cuda.empty_cache()

                self.model.to("cpu")
            elif attack_method["attack"] == "Sign_flipping":
                alpho = -4
                for e in range(self.local_epoch):
                    for data, labels in self.dataloader:
                        data, labels = data.float().to(self.device), labels.long().to(self.device)

                        optimizer.zero_grad()
                        outputs = self.model(data)

                        loss = eval(self.criterion)()(outputs, labels)
                        loss.backward()  # 反向传播
                        optimizer.step()
                        if self.device == "cuda":
                            torch.cuda.empty_cache()
                local_model_vector = torch.nn.utils.parameters_to_vector(self.model.parameters())
                local_model_vector = local_model_vector * alpho
                torch.nn.utils.vector_to_parameters(local_model_vector, self.model.parameters())
                self.model.to("cpu")
            else:  # 正常训练
                for e in range(self.local_epoch):
                    for data, labels in self.dataloader:

                        data, labels = data.float().to(self.device), labels.long().to(self.device)

                        optimizer.zero_grad()
                        outputs = self.model(data)

                        loss = eval(self.criterion)()(outputs, labels)
                        loss.backward()  # 反向传播
                        optimizer.step()

                        if self.device == "cuda":
                            torch.cuda.empty_cache()

                self.model.to("cpu")








        else:  # 正常训练
            for e in range(self.local_epoch):
                for data, labels in self.dataloader:

                    data, labels = data.float().to(self.device), labels.long().to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(data)

                    loss = eval(self.criterion)()(outputs, labels)
                    loss.backward()  # 反向传播
                    optimizer.step()

                    if self.device == "cuda":
                        torch.cuda.empty_cache()

            self.model.to("cpu")

    def client_evaluate(self):  # 本地测试
        """Evaluate local model using local dataset (same as training set for convenience)."""
        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():  # 本地测试
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss += eval(self.criterion)()(outputs, labels).item()

                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()  # 因为一个一个样本测试，所以准确度也加和

                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        test_loss = test_loss / len(self.dataloader)
        test_accuracy = correct / len(self.data)

        message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> Test accuracy: {100. * test_accuracy:.2f}%\n"
        print(message, flush=True);
        logging.info(message)
        del message;
        gc.collect()

        return test_loss, test_accuracy
