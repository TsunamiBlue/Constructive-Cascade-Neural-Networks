import os
import numpy as np
import matplotlib.pyplot as plt
import pandas


def cascor_plotting():
    path = os.path.join("logs", "log12.txt")
    x = []
    y_loss = []
    y_neuron = []
    y_acc = []
    with open(path) as f:
        for line in f.readlines():
            line = line.strip()
            line = line.replace(' ', '')
            line = line.replace('(', ",")
            line = line.replace(')', "")
            arrs = line.split(",")
            print(arrs)

            x.append(int(arrs[0]))
            y_loss.append(float(arrs[1]))
            y_neuron.append(int(arrs[2]))
            y_acc.append(float(arrs[3]))

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, y_acc, 'r-')
    ax2.plot(x, y_neuron, 'b-')

    ax1.set_xlabel('epochs')
    ax1.set_ylabel('test accuracy', color='r')
    ax2.set_ylabel('hidden neurons', color='b')

    # plt.show()

    fig2, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, y_loss, 'g-')
    ax2.plot(x, y_neuron, 'b-')

    ax1.set_xlabel('epochs')
    ax1.set_ylabel('training loss', color='g')
    ax2.set_ylabel('hidden neurons', color='b')
    return y_acc
    # plt.show()


def init_plotting():
    path = os.path.join("logs", "log_i_3.txt")
    x = []
    y_loss = []
    y_acc = []
    with open(path) as f:
        for line in f.readlines():
            line = line.strip()
            line = line.replace(' ', '')
            line = line.replace('(', ",")
            line = line.replace(')', "")
            arrs = line.split(",")
            print(arrs)

            x.append(int(arrs[0]))
            y_loss.append(float(arrs[1]))
            y_acc.append(float(arrs[2]))

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, y_acc, 'r-')
    ax2.plot(x, y_loss, 'b-')

    ax1.set_xlabel('epochs')
    ax1.set_ylabel('test accuracy', color='r')
    ax2.set_ylabel('training loss', color='b')
    return y_acc
    # plt.show()


if __name__ == "__main__":
    y_a1 = cascor_plotting()
    y_a2 = init_plotting()

    x = np.array(range(90))+1

    fig, am = plt.subplots()
    am.plot(x, y_a1, 'r-')
    am.plot(x, y_a2, 'b-')
    am.legend(["CasCor","FCN"], loc='lower right')
    am.set_xlabel('epochs')
    am.set_ylabel('Accuracy')
    plt.show()
