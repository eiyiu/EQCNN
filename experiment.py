import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from data import GroundStateDataset
from models import EQCNN, HEA_QCNN

wires_list = [[0, 1, 2, 3, 4, 5, 6, 7, 8],
              [2, 4, 6, 8],
              [4, 8]]

train_batch_size = 2
test_batch_size = 16
num_epoch = 750
lr = 1e-3

trainset = GroundStateDataset(len(wires_list[0]), 12)
tau_states = trainset.states[5:7]
trainloader = DataLoader(trainset, batch_size=train_batch_size,
                         shuffle=True)

testset = GroundStateDataset(len(wires_list[0]), 500)
testloader = DataLoader(testset, batch_size=test_batch_size,
                        shuffle=False)

equiv_qcnn = EQCNN(wires_list)
qcnn = HEA_QCNN(wires_list)

def train_and_test(model, trainloader, testloader, num_epoch, lr,
                   tau_states):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    tqdm_epoch = tqdm(range(num_epoch))

    for _ in tqdm_epoch:

        total_loss = 0.
        total_num = 0.

        for state, label, _ in trainloader:

            optimizer.zero_grad()
            out = model(state)
            loss = F.binary_cross_entropy(out, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            total_loss += loss.item() * state.size(0)
            total_num += state.size(0)
        
        avg_loss = total_loss / total_num
        tqdm_epoch.set_description(f'Average loss: {avg_loss}')

    out = model(tau_states)
    tau = torch.mean(out)

    x_triv = torch.empty(0)
    y_triv = torch.empty(0)
    x_top = torch.empty(0)
    y_top = torch.empty(0)
    pred = torch.empty(0)
    truth = torch.empty(0)

    with torch.no_grad():

        for state, label, alpha in testloader:

            out = model(state)
            x_triv = torch.cat((x_triv, alpha[out > tau]))
            y_triv = torch.cat((y_triv, out[out > tau]))
            x_top = torch.cat((x_top, alpha[out <= tau]))
            y_top = torch.cat((y_top, out[out <= tau]))
            pred = torch.cat((pred, out))
            truth = torch.cat((truth, label))

    pred[pred > tau] = 1.
    pred[pred <= tau] = 0.
    accuracy = 1 - torch.sum(torch.abs(pred - truth)) / pred.size(0)
    plt.scatter(x_triv, y_triv, c='b', label='Trivial')
    plt.scatter(x_top, y_top, c='r', label='Topological')
    plt.vlines(1., 0., 1., colors='k')
    plt.legend()
    plt.xlim(0., 2.)
    plt.ylim(0., 1.)
    plt.xlabel(r'$J_2/J_1$')
    plt.ylabel(r'$\frac{\langle SWAP \rangle + 1}{2}$')
    plt.title(r'Accuracy = {acc}, $\tau$ = {tau}'.format(
        acc=accuracy, tau=tau
    ))
    plt.savefig(f'{model.__class__.__name__}.png')
    plt.close()

train_and_test(equiv_qcnn, trainloader, testloader, num_epoch, lr,
               tau_states)
train_and_test(qcnn, trainloader, testloader, num_epoch, lr, tau_states)
