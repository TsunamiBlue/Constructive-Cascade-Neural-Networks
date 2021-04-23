import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
# from add_neuron import addNeuron
data_csv_path = ["SM_HighBP","SM_Normal","SM_pneumonia","SM_SARS"]


def data_preprocessing(data_path):
    path = os.path.join("sars-cov-1")
    datasets = []
    test_datasets = []
    label = int(-1)
    for category in data_path:
        # set label
        label+=1
        current_path = os.path.join(path,category+".csv")
        with open(current_path) as f:
            idx = 0
            data = []
            test_data = []
            for line in f.readlines():
                idx+=1
                single_data_list = line.strip().split(',')
                single_data = np.array(single_data_list,dtype=np.float32)
                if idx < 975:
                    data.append(single_data)
                else:
                    test_data.append(single_data)
            data = torch.from_numpy(np.array(data))
            test_data = torch.from_numpy(np.array(test_data))

            labels = torch.full((data.shape[0],1),label,dtype=torch.long)
            test_labels = torch.full((test_data.shape[0],1),label,dtype=torch.long)

            current_dataset = Data.TensorDataset(data,labels)
            current_test_dataset = Data.TensorDataset(test_data,test_labels)

            datasets.append(current_dataset)
            test_datasets.append(current_test_dataset)

            # print(f"there are {idx} datapoints under label {label}.")
    ans_dataset = Data.ConcatDataset(datasets)
    ans_test_dataset = Data.ConcatDataset(test_datasets)

    ans_dataloader = Data.DataLoader(ans_dataset,shuffle=True,batch_size=10)
    test_dataloader = Data.DataLoader(ans_test_dataset,shuffle=True,batch_size=10)

    return ans_dataloader,test_dataloader


def correlation_loss(predictions, labels):
    # batch_size = predictions.size()[0]
    # print(predictions.shape)
    # print(labels.shape)
    # print()
    vp = predictions - torch.mean(predictions)
    vl = labels - torch.mean(labels)

    # cost = torch.sum(vp * vl) / (torch.sqrt(torch.sum(vp ** 2)) * torch.sqrt(torch.sum(vl ** 2)))
    cost = torch.mean(vp*vl) / (torch.std(predictions)*torch.std(labels))
    return -cost


# Neural Network
class Cascade_Network(nn.Module):
    def __init__(self, input_size, num_classes, input_hidden_layers, hidden_hidden_layers, hidden_output_layers):
        super().__init__()
        self.n_hidden_layers = 0
        self.input_size = input_size
        self.num_classes = num_classes

        self.fc1 = nn.Linear(input_size, num_classes)

        # Init number of n*(n+3)/2 layers for use
        self.input_hidden_layers = input_hidden_layers
        self.hidden_hidden_layers = hidden_hidden_layers
        self.hidden_output_layers = hidden_output_layers

        self.correlation_outL2 = None


    def forward(self, x):
        outL3_1 = self.fc1(x)  # part of L3 weights, correlation of input and output
        if self.n_hidden_layers == 0:
            return outL3_1

        H = list()  # store connections of input classes and all hidden units (L1 weights and part of L3 weights)

        # store the first connection (between input and first hidden unit)
        H.append(F.leaky_relu(self.input_hidden_layers['0'](x)))

        if self.n_hidden_layers == 1:
            # Get connections related to L2 Weights
            outL2 = self.hidden_output_layers['0'](H[0])
            self.correlation_outL2 = outL2
            return outL3_1 + outL2

        # if n_hidden_layers>1, do the following iteration
        count1 = 0  # record the index of hidden_hidden_layers
        for i in range(1, self.n_hidden_layers):
            # build the current hidden unit, init with self.input_hidden
            current_hidden_unit = F.leaky_relu(self.input_hidden_layers[str(i)](x))

            for h in H:
                current_hidden_unit += F.leaky_relu(self.hidden_hidden_layers[str(count1)](h))
                count1 += 1
            H.append(current_hidden_unit)

        # Connect hidden unit to output
        total_out = outL3_1
        count2 = 0  # record the index of hidden_output_layers
        for h in H:
            total_out = total_out + self.hidden_output_layers[str(count2)](h)
            count2 += 1

        self.correlation_outL2 = self.hidden_output_layers[str(self.n_hidden_layers-1)](H[-1])
        return total_out

    def add_neuron(self):
        """
        When this function been called, the Casper will be added a new Hidden Neuron.
        This funciton takes the advantages of flexible of Pytorch. ModuleDict() is used to transfer out built layers
        to original network.
        :return: optimizer for new training network which has added a new Hidden Neuron
        """
        self.n_hidden_layers += 1
        self.input_hidden_layers[str(len(self.input_hidden_layers))] = nn.Linear(self.input_size, 1, bias=False)
        for n_connection in range(self.n_hidden_layers - 1):
            self.hidden_hidden_layers[str(len(self.hidden_hidden_layers))] = nn.Linear(1, 1,
                                                                                     bias=False)  # add bias for new neurons
        self.hidden_output_layers[str(len(self.hidden_output_layers))] = nn.Linear(1, self.num_classes, bias=False)



        # '''
        # Set different learning rate to different layers!
        # - Region L1: Weights connec/ng to new neuron.
        # – Region L2: Weights connected from new neuron to output neurons.
        # – Region L3: Remaining weights (all weights connected to and coming from the old hidden and input neurons).
        # L1>>L2>L3
        # The value of L1, L2 and L3 are 0.2, 0.005 and 0.001. Refer to the technique paper
        # '''
        # L1_params = list(map(id, self.input_hidden_layers[str(len(self.input_hidden_layers) - 1)].parameters()))  # L1
        # L2_params = list()
        # L2_params += list(map(id, self.hidden_output_layers[str(len(self.hidden_output_layers) - 1)].parameters()))  # L2
        # for num in range(self.n_hidden_layers - 1):
        #     L2_params += list(
        #         map(id, self.hidden_hidden_layers[str(len(self.hidden_hidden_layers) - num - 1)].parameters()))  # L2
        # L1L2 = L1_params + L2_params
        #
        # base_params = filter(lambda p: id(p) not in L1L2,
        #                      self.parameters())  # L3
        # params = [
        #     {'params': base_params, 'lr': 0.001},  # L3
        #     {'params': self.input_hidden_layers[str(len(self.input_hidden_layers) - 1)].parameters(), 'lr': 0.2},  # L1
        #     {'params': self.hidden_output_layers[str(len(self.hidden_output_layers) - 1)].parameters(), 'lr': 0.005},
        #     # L2
        # ]
        #
        # for num in range(self.n_hidden_layers - 1):
        #     params.append(
        #         {'params': self.hidden_hidden_layers[str(len(self.hidden_hidden_layers) - num - 1)].parameters(),
        #          'lr': 0.005}, )  # L2
        #
        # optimizer = torch.optim.RMSprop(params, momentum=0.9, weight_decay=0.00001, centered=True)
        # return optimizer
        return


    def optimize_correlation(self,dataloader,num_epochs=10, optimizer=None):
        """
        optimize the correlation between final output error of labels and internal output by new neuron.
        :param optimizer: use specified optimizer. default SGD
        :param num_epochs: number of sub-epochs. default 10
        :param dataloader: sub-data loader to train new input2hidden and hidden2hidden.
        :return: used optimizer
        """
        print("Start Correlation optimizing...")
        if optimizer is None:
            optimizer = optim.SGD(self.parameters(),lr=0.001,momentum=0.9)
        loss_sub_log = []
        for epoch in range(num_epochs):
            current_loss = float(0)
            batch_num = 0
            for batch_idx, batch_data in enumerate(dataloader,start=0):
                data, labels = batch_data
                optimizer.zero_grad()
                forward_correlation_result = self.forward(data)
                error = forward_correlation_result-labels
                loss = correlation_loss(self.correlation_outL2,error)
                loss.backward()
                optimizer.step()
                current_loss += loss.item()
                batch_num+=1
            loss_sub_log.append(current_loss/batch_num)
            print(f"sub epoch {epoch} correlation loss: {-current_loss/batch_num}")

        return optimizer


    def freeze_neuron(self,optimizer):
        """
        freeze the previous and current weight.
        :param optimizer: optimizer params to be frozen
        :return: optimizer: frozen optimizer
        """
        # optimizer = optim.SGD([
        #     {'params':self.input_hidden_layers,'lr':0},
        #     {'params':self.hidden_hidden_layers,'lr':0},
        #     {'params': self.hidden_output_layers}
        # ],
        #     lr=0.001,
        #     momentum=0.9)
        '''
        Set different learning rate to different layers!
        - Region L1: Weights connec/ng to new neuron.
        – Region L2: Weights connected from new neuron to output neurons.
        – Region L3: Remaining weights (all weights connected to and coming from the old hidden and input neurons).
        L1>>L2>L3
        The value of L1, L2 and L3 are 0.2, 0.005 and 0.001. Refer to the technique paper
        '''
        # L1_params = list(map(id, self.input_hidden_layers[str(len(self.input_hidden_layers) - 1)].parameters()))  # L1
        # L2_params = list()
        # L2_params += list(map(id, self.hidden_output_layers[str(len(self.hidden_output_layers) - 1)].parameters()))  # L2
        # for num in range(self.n_hidden_layers - 1):
        #     L2_params += list(
        #         map(id, self.hidden_hidden_layers[str(len(self.hidden_hidden_layers) - num - 1)].parameters()))  # L2
        # L1L2 = L1_params + L2_params
        #
        # base_params = filter(lambda p: id(p) not in L1L2,
        #                      self.parameters())  # L3
        # previous_params = optimizer.params
        # params = [
        #     {'params': self.input_hidden_layers[str(len(self.hidden_hidden_layers) - 1)].parameters(), 'lr': 0},  # hidden2hidden
        #     {'params': self.input_hidden_layers[str(len(self.input_hidden_layers) - 1)].parameters(), 'lr': 0},  # input2hidden
        #     {'params': self.hidden_output_layers[str(len(self.hidden_output_layers) - 1)].parameters(), 'lr': 0.001}, # output2hidden
        # ]
        n_neurons = self.n_hidden_layers
        params = []
        for i in range(n_neurons):
            params.append(
                # input2hidden
                {'params': self.input_hidden_layers[str(i)].parameters(), 'lr': 0},
            )
            params.append(
                # hidden2output
                {'params': self.hidden_output_layers[str(i)].parameters(), 'lr': 0.001},
            )
        if n_neurons > 1:
            for i in range(int(n_neurons*(n_neurons-1)/2)):
                params.append(
                    # hidden2hidden
                    {'params': self.hidden_hidden_layers[str(i)].parameters(), 'lr': 0},
                )
        # print(params)
        optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=0.00001,lr=0.001)
        return optimizer





# if __name__ == "__main__":
#     train_dataloader, test_dataloader = data_preprocessing(data_csv_path)
#     sample = train_dataloader.dataset
#     print(train_dataloader.dataset.__len__())
#     print(np.array(list(enumerate(train_dataloader.dataset))).shape)
#     # print(np.array(list(enumerate(dataloader.dataset)))[0][1])
#
#     input_hidden_layers = nn.ModuleDict()
#     hidden_hidden_layers = nn.ModuleDict()
#     hidden_output_layers = nn.ModuleDict()
#
#     cascade_network = Cascade_Network(23,4,input_hidden_layers,hidden_hidden_layers,hidden_output_layers)
#     print(cascade_network)
#
#     loss_CE=nn.CrossEntropyLoss()
#     optimizer = optim.SGD(
#         cascade_network.parameters(),
#         lr=0.001,
#         momentum=0.9)
#     print(cascade_network)
#
#
#     loss_log = []
#     loss_epoch_log = []
#     hidden_neuron_num = 0
#     for epoch in range(10):
#         current_loss = float(0)
#         accumulate_loss = float(0)
#         for batch_idx, batch_data in enumerate(train_dataloader,start=0):
#
#             data,labels = batch_data
#             optimizer.zero_grad()
#             forward_result = cascade_network(data)
#             loss = loss_CE(forward_result,labels.squeeze())
#             loss.backward()
#             optimizer.step()
#             current_loss += loss.item()
#
#             if batch_idx %100 == 99:
#                 print(f"epoch {epoch+1} batch No.{batch_idx+1} loss: {current_loss/100}")
#                 loss_log.append(current_loss/100)
#                 accumulate_loss = current_loss/100
#                 current_loss = 0
#
#         if loss_epoch_log != [] and loss_epoch_log[-1] - accumulate_loss > 0.005:
#             cascade_network.add_neuron()
#             hidden_neuron_num += 1
#             print(f"ADD ONE NEURON in epoch {epoch}.")
#
#
#         loss_epoch_log.append(accumulate_loss)
#
#     # final test set
#     true_postive = 0
#     total = 0
#     for batch_idx, batch_data in enumerate(test_dataloader,start=0):
#         data,labels = batch_data
#         prediction = cascade_network(data)
#         ans = torch.tensor([np.argmax(F.softmax(each).detach().numpy()) for each in prediction])
#         # print(ans)
#         # print(labels.squeeze())
#         labels = labels.squeeze()
#         for i in range(ans.shape[0]):
#             if ans[i]==labels[i]:
#                 true_postive+=1
#             total+=1
#
#
#     print(f"Final test accuracy: {true_postive*100/total} %")
#     print(f"ratio: {true_postive}/{total}")
#     print(f"overall hidden neuron added: {hidden_neuron_num}")
#     print("DONE.")
