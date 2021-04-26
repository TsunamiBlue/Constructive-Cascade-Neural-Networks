import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F

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


class Cascade_Network(nn.Module):

    def __init__(self,max_used_cascade=10):
        super(Cascade_Network,self).__init__()
        self.input_fcn = nn.Linear(23,10)
        self.output_fcn = nn.Linear(10,4)

        self.used_neuron = 0
        self.cascade_layers = []



    def forward(self,x):
        # x = F.relu(self.input_fcn(x))
        # x = self.output_fcn(x)
        # x = F.softmax(x,dim=1)

        x = F.relu(self.input_fcn(x))
        x = F.softmax(x, dim=1)
        x_hidden = []
        if self.used_casacade != 0:
            for idx, layer in enumerate(self.cascade_layers):

                    tmp = self.cascade_layers[idx](x)


        return x


    def add_neuron(self):

        self.cascade_layers.append(nn.Linear(10*(self.used_casacade+1),10))
        self.used_casacade +=1

        return






if __name__ == "__main__":
    train_dataloader, test_dataloader = data_preprocessing(data_csv_path)
    sample = train_dataloader.dataset
    print(train_dataloader.dataset.__len__())
    print(np.array(list(enumerate(train_dataloader.dataset))).shape)
    # print(np.array(list(enumerate(dataloader.dataset)))[0][1])
    cascade_network = Cascade_Network(max_used_cascade=3)
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
            forward_result = cascade_network(data,1)
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
        prediction = cascade_network(data,1)
        ans = torch.tensor([np.argmax(each.detach().numpy()) for each in prediction])
        # print(ans)
        # print(labels.squeeze())
        labels = labels.squeeze()
        for i in range(ans.shape[0]):
            if ans[i]==labels[i]:
                true_postive+=1
            total+=1
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

    print(f"Final test accuracy: {true_postive*100/total} %")
    print(f"tp / total: {true_postive}/{total}")
    print("DONE.")


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