import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
from cascor_network import Cascade_Network, data_preprocessing




data_csv_path = ["SM_HighBP","SM_Normal","SM_pneumonia","SM_SARS"]


if __name__ == "__main__":
    train_dataloader, test_dataloader = data_preprocessing(data_csv_path)
    sample = train_dataloader.dataset
    # print(train_dataloader.dataset.__len__())
    print(np.array(list(enumerate(train_dataloader.dataset))).shape)

    input_hidden_layers = nn.ModuleDict()
    hidden_hidden_layers = nn.ModuleDict()
    hidden_output_layers = nn.ModuleDict()

    cascade_network = Cascade_Network(23,4,input_hidden_layers,hidden_hidden_layers,hidden_output_layers)
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

    n_epochs = 90
    max_hidden = 10

    for epoch in range(n_epochs):
        current_loss = float(0)
        epoch_loss = float(0)
        for batch_idx, batch_data in enumerate(train_dataloader,start=0):

            data,labels = batch_data
            optimizer.zero_grad()
            forward_result = F.softmax(cascade_network(data),dim=1)
            loss = loss_CE(forward_result,labels.squeeze())
            loss.backward()
            optimizer.step()
            current_loss += loss.item()

            if batch_idx %100 == 99:
                current_loss /= 100
                print(f"epoch {epoch+1} batch No.{batch_idx+1} loss: {current_loss}")
                epoch_loss = current_loss
                current_loss = 0

        #
        if loss_epoch_log != [] and add_neuron_counter == 0 and previous_loss - epoch_loss < 0 \
            and epoch < n_epochs*0.8 and cascade_network.num_hiddens < max_hidden:

            cascade_network.add_neuron()
            hidden_neuron_num += 1

            add_neuron_counter = 5

            print(f"ADD NEURON in epoch {epoch}.\n There are {cascade_network.num_hiddens} in total")
            cascade_network.optimize_correlation(train_dataloader)
            optimizer = cascade_network.freeze_neuron(optimizer)
            print(f"ADD {hidden_neuron_num}th NEURON ends")

        previous_loss = epoch_loss
        loss_epoch_log.append(epoch_loss)
        add_neuron_counter -= 1
        add_neuron_counter = max(add_neuron_counter,0)



    # final test set
    true_postive = 0
    total = 0
    for batch_idx, batch_data in enumerate(test_dataloader,start=0):
        data,labels = batch_data
        prediction = F.softmax(cascade_network(data),dim=1)
        # print(prediction.shape)
        ans = torch.tensor([np.argmax(each.detach().numpy()) for each in prediction])
        print("answer vs label")
        print(ans)
        print(labels.squeeze())
        labels = labels.squeeze()
        for i in range(ans.shape[0]):
            if ans[i]==labels[i]:
                true_postive+=1
            total+=1


    print(f"Final test accuracy: {true_postive*100/total} %")
    print(f"ratio: {true_postive}/{total}")
    print(f"overall hidden neuron added: {hidden_neuron_num}")
    print("DONE.")
