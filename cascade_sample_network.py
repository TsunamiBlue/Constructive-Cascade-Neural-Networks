import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
from add_neuron import addNeuron
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

            print(f"there are {idx} datapoints under label {label}.")
    ans_dataset = Data.ConcatDataset(datasets)
    ans_test_dataset = Data.ConcatDataset(test_datasets)

    ans_dataloader = Data.DataLoader(ans_dataset,shuffle=True,batch_size=10)
    test_dataloader = Data.DataLoader(ans_test_dataset,shuffle=True,batch_size=10)

    return ans_dataloader,test_dataloader


# Neural Network
class Cascade_Network(nn.Module):
    def __init__(self, input_size, num_classes, input_hidden_layers, hidden_hidden_layers, hidden_output_layers):
        super().__init__()
        self.n_hidden_layers = 0
        self.fc1 = nn.Linear(input_size, num_classes)
        # self.bn_input = nn.BatchNorm1d(10, momentum=0.9)

        # Init number of n*(n+3)/2 layers for use
        self.input_hidden_layers = input_hidden_layers
        self.hidden_hidden_layers = hidden_hidden_layers
        self.hidden_output_layers = hidden_output_layers

    def forward(self, x):
        outL3_1 = self.fc1(x)  # part of L3 weights, correlation of input and output
        if self.n_hidden_layers == 0:
            return outL3_1

        H = list()  # store connections of input classes and all hidden units (L1 weights and part of L3 weights)

        # store the first connection (between input and first hidden unit)
        H.append(F.leaky_relu(self.input_hidden_layers['0'](x)))

        if self.n_hidden_layers == 1:
            for h in H:
                # Get connections related to L2 Weights
                outL2 = self.hidden_output_layers['0'](h)
                return outL3_1 + outL2

        # if n_hidden_layers>1, do the following iteration
        count1 = 0  # record the index of hidden_hidden_layers
        for i in range(1, self.n_hidden_layers):
            # build the current hidden unit, init with self.input_hidden
            current_hidden_unit = F.leaky_relu(self.input_hidden_layers[str(i)](x))
            c_list = list()
            c_list.append(current_hidden_unit)
            for h in H:
                # if len(H)-count1 > 3:
                #     previous_connection.detach()
                current_hidden_unit += F.leaky_relu(self.hidden_hidden_layers[str(count1)](h))
                count1 += 1
            H.append(current_hidden_unit)

        # Connect hidden unit to output
        total_out = outL3_1
        count2 = 0  # record the index of hidden_output_layers
        for h in H:
            total_out = total_out + self.hidden_output_layers[str(count2)](h)
            count2 += 1
        return total_out






if __name__ == "__main__":
    train_dataloader, test_dataloader = data_preprocessing(data_csv_path)
    sample = train_dataloader.dataset
    print(train_dataloader.dataset.__len__())
    print(np.array(list(enumerate(train_dataloader.dataset))).shape)
    # print(np.array(list(enumerate(dataloader.dataset)))[0][1])

    input_hidden_layers = nn.ModuleDict()
    hidden_hidden_layers = nn.ModuleDict()
    hidden_output_layers = nn.ModuleDict()

    cascade_network = Cascade_Network(23,4,input_hidden_layers,hidden_hidden_layers,hidden_output_layers)
    addNeuron(cascade_network)

    loss_CE=nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        cascade_network.parameters(),
        lr=0.001,
        momentum=0.9)
    print(cascade_network)


    loss_log = []
    for epoch in range(10):
        current_loss = float(0)
        for batch_idx, batch_data in enumerate(train_dataloader,start=0):

            data,labels = batch_data
            optimizer.zero_grad()
            forward_result = cascade_network(data)
            loss = loss_CE(forward_result,labels.squeeze())
            loss.backward()
            optimizer.step()
            current_loss += loss.item()

            if batch_idx %100 == 99:
                print(f"epoch {epoch+1} batch No.{batch_idx+1} loss: {current_loss/100}")
                loss_log.append(current_loss/100)
                current_loss = 0

    true_postive = 0
    total = 0
    for batch_idx, batch_data in enumerate(test_dataloader,start=0):
        data,labels = batch_data
        prediction = cascade_network(data)
        ans = torch.tensor([np.argmax(each.detach().numpy()) for each in prediction])
        # print(ans)
        # print(labels.squeeze())
        labels = labels.squeeze()
        for i in range(ans.shape[0]):
            if ans[i]==labels[i]:
                true_postive+=1
            total+=1


    print(f"Final test accuracy: {true_postive*100/total} %")
    print(f"ratio: {true_postive}/{total}")
    print("DONE.")



