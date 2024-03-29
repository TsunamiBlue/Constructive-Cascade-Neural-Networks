import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
from cascor_network import test_accuracy

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
                single_data = np.array(single_data_list,dtype=np.float32)[:12]
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

    ans_dataloader = Data.DataLoader(ans_dataset,shuffle=True,batch_size=512)
    test_dataloader = Data.DataLoader(ans_test_dataset,shuffle=True,batch_size=512)

    return ans_dataloader,test_dataloader


class Cascade_Network(nn.Module):

    def __init__(self):
        super(Cascade_Network,self).__init__()
        self.fcn = nn.Linear(12,4,bias=True)
        # self.cascade_layer = nn.Linear(10,10)
        # self.output_fcn = nn.Linear(10,4)
        # self.hidden_layer = None

        # self.cascade_layers = []
        # for i in range(10):
        #     self.cascade_layers.append(nn.Linear(10,10))



    def forward(self,x):
        x = self.fcn(x)
        # x = F.softmax(x,dim=1)
        return x






if __name__ == "__main__":
    train_dataloader, test_dataloader = data_preprocessing(data_csv_path)
    sample = train_dataloader.dataset
    print(train_dataloader.dataset.__len__())
    print(np.array(list(enumerate(train_dataloader.dataset))).shape)
    # print(np.array(list(enumerate(dataloader.dataset)))[0][1])
    cascade_network = Cascade_Network()
    loss_CE=nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        cascade_network.parameters(),
        lr=0.001,
        momentum=0.9)
    print(cascade_network)

    loss_log = []
    for epoch in range(90):
        # current_loss = float(0)
        for batch_idx, batch_data in enumerate(train_dataloader,start=0):

            data,labels = batch_data
            optimizer.zero_grad()
            forward_result = cascade_network(data)
            loss = loss_CE(forward_result,labels.squeeze())
            loss.backward()
            optimizer.step()
            current_loss = loss.item()

            # if batch_idx %100 == 99:
            #     print(f"epoch {epoch+1} batch No.{batch_idx+1} loss: {current_loss/100}")
            #     loss_log.append(current_loss/100)
            #     current_loss = 0
        tp, total = test_accuracy(test_dataloader, cascade_network)
        print(f"epoch {epoch + 1} test acc: {tp * 100 / total} %")
        loss_log.append((current_loss,tp * 100 / total))

    # true_postive = 0
    # total = 0
    # for batch_idx, batch_data in enumerate(test_dataloader,start=0):
    #     data,labels = batch_data
    #     prediction = cascade_network(data)
    #     ans = torch.tensor([np.argmax(each.detach().numpy()) for each in prediction])
    #     # print(ans,int(labels.detach().numpy()[0][0]))
    #     labels = labels.squeeze()
    #     for i in range(ans.shape[0]):
    #         if ans[i] == labels[i]:
    #             true_postive += 1
    #         total += 1
    final_true_positive, final_total = test_accuracy(test_dataloader, cascade_network)

    print(f"Additional Loss Log \n  Loss Log Format: epoch, loss, number of hidden neurons, accuracy")
    for i, item in enumerate(loss_log):
        print(i + 1, item)

    print(f"Final test accuracy: {final_true_positive*100/final_total} %")
    print(f"ratio: {final_true_positive}/{final_total}")
    print("DONE.")