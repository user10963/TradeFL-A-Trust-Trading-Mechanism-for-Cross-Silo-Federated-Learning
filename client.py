import torch
from .models import *


class Client(object):

    def __init__(self, conf, model, train_dataset, id=-1):

        self.conf = conf

        self.local_model = get_model(self.conf["model_name"])

        self.client_id = id

        self.train_dataset = train_dataset

        all_range = list(range(len(self.train_dataset)))

        data_len = int(self.conf['self_data'] * self.conf['compute_power'][id])

        if data_len % self.conf['batch_size'] == 1:
            data_len -= 1
        train_indices = all_range[id * self.conf['self_data']
            :id * self.conf['self_data'] + data_len]
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"],
                                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                            train_indices))

    def local_train(self, model):

        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
                                    momentum=self.conf['momentum'])

        self.local_model.train()
        for e in range(self.conf["client_epoch_num"]):

            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch

                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()
                optimizer.zero_grad()
                output = self.local_model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - model.state_dict()[name])

        return diff
