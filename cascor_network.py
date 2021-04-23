import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F


data_csv_path = ["SM_HighBP","SM_Normal","SM_pneumonia","SM_SARS"]


def data_preprocessing(data_path,num_feature=23):
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
                single_data = np.array(single_data_list,dtype=np.float32)[:num_feature]
                if idx < 924:
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

    ans_dataloader = Data.DataLoader(ans_dataset,shuffle=True,batch_size=50)
    test_dataloader = Data.DataLoader(ans_test_dataset,shuffle=True,batch_size=10)

    return ans_dataloader,test_dataloader


def correlation_loss(predictions, labels):
    """
    A simple correlation loss function.
    :param predictions: Tensor
    :param labels: Tensor
    :return: ordinary correlation loss
    """

    vp = predictions - torch.mean(predictions)
    vl = labels - torch.mean(labels)

    # cost = torch.sum(vp * vl) / (torch.sqrt(torch.sum(vp ** 2)) * torch.sqrt(torch.sum(vl ** 2)))
    cost = torch.mean(vp*vl) / (torch.std(predictions)*torch.std(labels))
    return cost


# Constructive Cascade Neural Network
class Cascade_Network(nn.Module):
    def __init__(self, input_size, num_classes, input2hidden_layers, hidden2hidden_layers, hidden2output_layers):
        super().__init__()
        self.num_hiddens = 0
        self.input_size = input_size
        self.num_classes = num_classes

        self.initial_input_layer = nn.Linear(input_size, num_classes)

        # module dict for different layers
        self.input2hidden_layers = input2hidden_layers
        self.hidden2hidden_layers = hidden2hidden_layers
        self.hidden2output_layers = hidden2output_layers

        # for correlation GD
        self.latest_hidden_out = None


    def forward(self, x):
        input_out = self.initial_input_layer(x)  # input directly to output
        if self.num_hiddens == 0:
            return input_out
        # store all outputs from input layer which are also the input for any hidden layers
        H_in = list()
        # store the first output from input layer (between input and first hidden unit)
        H_in.append(F.leaky_relu(self.input2hidden_layers['0'](x)))

        if self.num_hiddens == 1:
            # if only one hidden layer inserted
            out2 = self.hidden2output_layers['0'](H_in[0])
            self.latest_hidden_out = out2
            return input_out + out2

        # if num_hiddens>1, do the following iteration
        hidden_idx = 0  # record the index of hidden2hidden_layers
        for i in range(1, self.num_hiddens):
            # build the current hidden unit, init with self.input_hidden
            current_hidden_unit = F.leaky_relu(self.input2hidden_layers[str(i)](x))

            for h in H_in:
                current_hidden_unit += F.leaky_relu(self.hidden2hidden_layers[str(hidden_idx)](h))
                hidden_idx += 1
            H_in.append(current_hidden_unit)

        # Connect hidden layer to output
        final_out = input_out
        hidden2out_idx = 0  # record the index of hidden2output_layers
        for h in H_in:
            final_out = final_out + self.hidden2output_layers[str(hidden2out_idx)](h)
            hidden2out_idx += 1

        self.latest_hidden_out = self.hidden2output_layers[str(self.num_hiddens-1)](H_in[-1])
        return final_out

    def add_neuron(self):
        """
        the network will add one more neuron/layer into hidden layer.
        NOTICE: call optimize_correlation and freeze_neuron afterwards to get the optimized frozen weights and freeze
                the input2hidden and hidden2hidden layer.
        """
        self.num_hiddens += 1
        self.input2hidden_layers[str(len(self.input2hidden_layers))] = nn.Linear(self.input_size, 1, bias=False)
        for n_connection in range(self.num_hiddens - 1):
            self.hidden2hidden_layers[str(len(self.hidden2hidden_layers))] = nn.Linear(1, 1,bias=False)
        self.hidden2output_layers[str(len(self.hidden2output_layers))] = nn.Linear(1, self.num_classes, bias=False)

        return


    def optimize_correlation(self,dataloader,num_epochs=10, optimizer=None):
        """
        optimize the correlation between final output error of labels and latest internal output by new neuron.
        :param optimizer: use specified optimizer. default SGD
        :param num_epochs: number of sub-epochs. default 10
        :param dataloader: sub-data loader to train new input2hidden and hidden2hidden.
        :return: used optimizer
        """
        print("    Start Correlation optimizing...")
        if optimizer is None:
            optimizer = optim.SGD(self.parameters(),lr=0.001,momentum=0.9)
        loss_sub_log = []
        for epoch in range(num_epochs):
            current_loss = float(0)
            # batch_num = 0
            for batch_idx, batch_data in enumerate(dataloader,start=0):
                data, labels = batch_data
                optimizer.zero_grad()
                forward_correlation_result = F.softmax(self.forward(data),dim=1)
                # labels_extended = labels.expand(forward_correlation_result.shape)
                labels_extended = torch.zeros(forward_correlation_result.shape)
                for idx, label in enumerate(labels):
                    labels_extended[idx][label] = 1

                error = forward_correlation_result-labels_extended
                # print(forward_correlation_result[0])
                # print(labels_extended[0])
                # print(labels[0])
                loss = -correlation_loss(self.latest_hidden_out,error)
                loss.backward()
                optimizer.step()
                current_loss = loss.item()
                # batch_num+=1
            loss_sub_log.append(current_loss)
            print(f"    sub epoch {epoch} correlation loss: {-current_loss}")

        return optimizer


    def freeze_neuron(self,optimizer):
        """
        freeze the previous and current weight.
        :param optimizer: optimizer params to be frozen
        :return: optimizer: frozen optimizer
        """

        n_neurons = self.num_hiddens
        params = []
        for i in range(n_neurons):
            params.append(
                # input2hidden
                {'params': self.input2hidden_layers[str(i)].parameters(), 'lr': 0},
            )
            params.append(
                # hidden2output
                {'params': self.hidden2output_layers[str(i)].parameters(), 'lr': 0.001},
            )
        if n_neurons > 1:
            for i in range(int(n_neurons*(n_neurons-1)/2)):
                params.append(
                    # hidden2hidden
                    {'params': self.hidden2hidden_layers[str(i)].parameters(), 'lr': 0},
                )
        # print(params)
        # self.parameters = params
        optimizer = torch.optim.SGD(params, momentum=0.9,lr=0.001)
        return optimizer


def test_accuracy(dataloader,network):
    # final test set
    true_postive = 0
    total = 0
    for batch_idx, batch_data in enumerate(dataloader, start=0):
        data, labels = batch_data
        prediction = F.softmax(network(data), dim=1)
        # print(prediction.shape)
        ans = torch.tensor([np.argmax(each.detach().numpy()) for each in prediction])
        # print("answer vs label")
        # print(ans)
        # print(labels.squeeze())
        labels = labels.squeeze()
        for i in range(ans.shape[0]):
            if ans[i] == labels[i]:
                true_postive += 1
            total += 1
    return true_postive, total


if __name__ == "__main__":

    num_feature = 12
    n_epochs = 90
    max_hidden = 10


    train_dataloader, test_dataloader = data_preprocessing(data_csv_path,num_feature=num_feature)
    sample = train_dataloader.dataset
    # print(train_dataloader.dataset.__len__())
    print(np.array(list(enumerate(train_dataloader.dataset))).shape)

    input_hidden_layers = nn.ModuleDict()
    hidden_hidden_layers = nn.ModuleDict()
    hidden_output_layers = nn.ModuleDict()

    cascade_network = Cascade_Network(num_feature,4,input_hidden_layers,hidden_hidden_layers,hidden_output_layers)
    print(cascade_network)

    loss_CE=nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        cascade_network.parameters(),
        lr=0.001,
        momentum=0.9)

    loss_epoch_log = []
    hidden_neuron_num = 0
    # limit the frequency of additional neurons
    add_neuron_counter = 0
    previous_loss = float('inf')



    for epoch in range(n_epochs):
        current_loss = float(0)
        epoch_loss = float(0)
        for batch_idx, batch_data in enumerate(train_dataloader,start=0):
            data,labels = batch_data
            optimizer.zero_grad()
            forward_result = cascade_network(data)
            loss = loss_CE(forward_result,labels.squeeze())
            loss.backward()
            optimizer.step()
            current_loss = loss.item()
            epoch_loss = current_loss

            # if batch_idx %100 == 99:
            #     current_loss /= 100
            #     print(f"epoch {epoch+1} batch No.{batch_idx+1} loss: {current_loss}")
            #     epoch_loss = current_loss
            #     current_loss = 0
        print(f"epoch {epoch+1} training loss: {current_loss}")
        #
        if loss_epoch_log != [] and add_neuron_counter == 0 and previous_loss - epoch_loss < 0 \
                and epoch < n_epochs*0.8 and cascade_network.num_hiddens < max_hidden:

            cascade_network.add_neuron()
            hidden_neuron_num += 1
            add_neuron_counter = 5

            print(f"ADD NEURON in epoch {epoch}. There are {cascade_network.num_hiddens} in total")
            cascade_network.optimize_correlation(train_dataloader)
            optimizer = cascade_network.freeze_neuron(optimizer)
            # print(f"ADD {hidden_neuron_num}th NEURON ends")

        previous_loss = epoch_loss
        add_neuron_counter -= 1
        add_neuron_counter = max(add_neuron_counter,0)

        tp,total = test_accuracy(test_dataloader,cascade_network)
        print(f"epoch {epoch+1} test acc: {tp*100/total} %")

        loss_epoch_log.append((epoch_loss,cascade_network.num_hiddens,tp*100/total))




    # # final test set
    # true_postive = 0
    # total = 0
    # for batch_idx, batch_data in enumerate(test_dataloader,start=0):
    #     data,labels = batch_data
    #     prediction = F.softmax(cascade_network(data),dim=1)
    #     # print(prediction.shape)
    #     ans = torch.tensor([np.argmax(each.detach().numpy()) for each in prediction])
    #     print("answer vs label")
    #     print(ans)
    #     print(labels.squeeze())
    #     labels = labels.squeeze()
    #     for i in range(ans.shape[0]):
    #         if ans[i]==labels[i]:
    #             true_postive+=1
    #         total+=1
    final_true_positive, final_total = test_accuracy(test_dataloader,cascade_network)


    print(f"Final test accuracy: {final_true_positive*100/final_total} %")
    print(f"ratio: {final_true_positive}/{final_total}")
    print(f"overall hidden neuron added: {hidden_neuron_num}")
    print("DONE.")

    for i,item in enumerate(loss_epoch_log):
        print(i,item)



# if __name__ == "__main__":
#     train_dataloader, test_dataloader = data_preprocessing(data_csv_path)
#     sample = train_dataloader.dataset
#     print(train_dataloader.dataset.__len__())
#     print(np.array(list(enumerate(train_dataloader.dataset))).shape)
#     # print(np.array(list(enumerate(dataloader.dataset)))[0][1])
#
#     input2hidden_layers = nn.ModuleDict()
#     hidden2hidden_layers = nn.ModuleDict()
#     hidden2output_layers = nn.ModuleDict()
#
#     cascade_network = Cascade_Network(23,4,input2hidden_layers,hidden2hidden_layers,hidden2output_layers)
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



