import torch
import torch.nn as nn
import pennylane as qp

n_qubits = 9
dev = qp.device('default.qubit', wires=n_qubits)

@qp.qnode(dev, interface='torch')
def equiv_circuit(state, weights, wires_list):
    qp.StatePrep(state, wires=range(n_qubits))
    l = len(wires_list)
    for i in range(l - 1):
        num_wires = len(wires_list[i])
        if num_wires % 2 == 0:
            num_layers = 3 * num_wires // 2
        else:
            num_layers = (3 * num_wires - 1) // 2
        for ii in range(num_layers):
            for k in range(num_wires // 2):
                qp.exp(
                    qp.SWAP([wires_list[i][2 * k],
                             wires_list[i][2 * k + 1]]),
                    -1j * weights[i][2 * ii]
                )
            if num_wires % 2 == 0:
                for m in range(num_wires // 2 - 1):
                    qp.exp(
                        qp.SWAP([wires_list[i][2 * m + 1],
                                 wires_list[i][2 * m + 2]]),
                        -1j * weights[i][2 * ii + 1]
                    )
                qp.exp(
                    qp.SWAP([wires_list[i][-1],
                             wires_list[i][0]]),
                    -1j * weights[i][2 * ii + 1]
                )
            else:
                for m in range(num_wires // 2):
                    qp.exp(
                        qp.SWAP([wires_list[i][2 * m + 1],
                                 wires_list[i][2 * m + 2]]),
                        -1j * weights[i][2 * ii + 1]
                    )
        for wire in wires_list[i]:
            if wire not in wires_list[i + 1]:
                qp.measure(wire)
    qp.exp(qp.SWAP(wires_list[-1]), -1j * weights[-1])
    return qp.density_matrix(wires_list[-1])
    
@qp.qnode(dev, interface='torch')
def circuit(state, weights, wires_list):
    qp.StatePrep(state, wires=range(n_qubits))
    l = len(wires_list)
    for i in range(l - 1):
        for j in range(3):
            qp.BasicEntanglerLayers(weights[i][j], wires=wires_list[i])
        for wire in wires_list[i]:
            if wire not in wires_list[i + 1]:
                qp.measure(wire)
    qp.BasicEntanglerLayers(weights[-1], wires=wires_list[-1])
    return qp.density_matrix(wires_list[-1])


class EQCNN(nn.Module):
    def __init__(self, wires_list):
        super().__init__()
        li = [
            nn.Parameter(torch.empty((3 * len(wires_list[i]) // 2) * 2,
                                     1))
            for i in range(len(wires_list) - 1)
        ]
        li.append(nn.Parameter(torch.empty(1)))
        self.weights = nn.ParameterList(li)
        self.wires_list = wires_list
        self.reset_parameters()
    
    def reset_parameters(self):
        for param in self.weights:
            nn.init.uniform_(param, a=-1 * torch.pi, b=torch.pi)
    
    def forward(self, state):
        rho = equiv_circuit(state, self.weights, self.wires_list)
        y = torch.matmul(rho.cfloat(),
                         torch.tensor([[1., 0., 0., 0.],
                                       [0., 0., 1., 0.],
                                       [0., 1., 0., 0.],
                                       [0., 0., 0., 1.]]).cfloat())
        y = (torch.sum(torch.diagonal(y, 0, -2, -1), -1)).real
        return (y + 1) / 2


class HEA_QCNN(nn.Module):
    def __init__(self, wires_list):
        super().__init__()
        li = [
            nn.Parameter(torch.empty(3, 1, len(wires_list[i])))
            for i in range(len(wires_list) - 1)
        ]
        li.append(nn.Parameter(torch.empty(1, 2)))
        self.weights = nn.ParameterList(li)
        self.wires_list = wires_list
        self.reset_parameters()
    
    def reset_parameters(self):
        for param in self.weights:
            nn.init.uniform_(param, a=-1 * torch.pi, b=torch.pi)
    
    def forward(self, state):
        rho = circuit(state, self.weights, self.wires_list)
        y = torch.matmul(rho.cfloat(),
                         torch.tensor([[1., 0., 0., 0.],
                                       [0., 0., 1., 0.],
                                       [0., 1., 0., 0.],
                                       [0., 0., 0., 1.]]).cfloat())
        y = (torch.sum(torch.diagonal(y, 0, -2, -1), -1)).real
        return (y + 1) / 2
