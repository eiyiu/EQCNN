import numpy as np
import pennylane as qp
import torch
from torch.utils.data import Dataset


class GroundStateDataset(Dataset):
    def __init__(self, N, num_samples):
        if num_samples % 2 == 1:
            raise ValueError("Number of samples must be even.")
        alpha = np.linspace(0, 2, num_samples + 2)
        alpha = alpha[1: -1]

        states_0 = []
        if N % 2 == 1:
            states_1 = []
        label = []
        for i in range(num_samples):
            J = np.ones((3, N, N))
            J[:, 1:: 2, :] = alpha[i]
            H = qp.spin.heisenberg('chain', [N], J / 4)
            _, v = np.linalg.eigh(H.matrix())
            states_0.append(v[:, 0])
            if N % 2 == 1:
                states_1.append(v[:, 1])
            if i < num_samples // 2:
                label.append(1.)
            else:
                label.append(0.)
        states_0 = np.vstack(states_0)
        if N % 2 == 1:
            states_1 = np.vstack(states_1)
            gaussian = np.random.normal(size=(num_samples, 4))
            s3 = gaussian / np.linalg.norm(gaussian, axis=1)[:, None]
            coeff_1 = s3[:, 0] + 1j * s3[:, 1]
            coeff_2 = s3[:, 2] + 1j * s3[:, 3]
            states_0 = (coeff_1[:, None] * states_0 +
                        coeff_2[:, None] * states_1)

        self.states = torch.tensor(states_0)
        self.label = torch.tensor(label)
        self.alpha = torch.tensor(alpha)
 
    def __len__(self):
        return len(self.alpha)
    
    def __getitem__(self, idx):
        return self.states[idx], self.label[idx], self.alpha[idx]
