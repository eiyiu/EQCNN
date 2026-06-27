# EQCNN
An implementation of the paper ["Theory for Equivariant Quantum Neural Networks"](https://doi.org/10.1103/PRXQuantum.5.020328) by Nguyen et al using PennyLane and PyTorch.

We recreate the experiment in the [paper](https://doi.org/10.1103/PRXQuantum.5.020328) and train classifiers to classify the ground states of the Hamiltonian 

$$ H_\alpha = J_1 \sum_{i \text{ odd}} S_i S_{i + 1} + J_2\sum_{i \text{ even}} S_i S_{i+1},$$

where $\alpha = J_2 / J_1$, $J_1 = 1$ and $i = 1,..., N-1$.  This Hamiltonian commutes with the diagonal action of $SU(2)$ and its ground energy is degenerate when the system size is odd.  When $\alpha < 1$, the system is in the trivial phase, and when $\alpha \geq 1$ the system is in the toplogically protected phase.

We generate ground states with different $\alpha \in [0,2]$ by using NumPy's eigensolver `eigh`.  For odd system sizes, since the ground energy is degenerate and so the ground state is not unique up to a global phase, we apply a random $SU(2)$ rotation to rotate the obtained ground state in the eigenspace.  The random rotation is sampled uniformly on $S^3 \simeq SU(2)$ by first sampling $X_1, ..., X_4 \sim \mathcal N(0,1)$ and normalizing them.

Since the system pocesses $SU(2)$ symmetry, the authors of the [paper](https://doi.org/10.1103/PRXQuantum.5.020328) demonstrate the usefulness of equivariant QNN by training a $SU(2)$-equivariant quantum convolutional neural network (EQCNN) and a non-equivariant quantum convolutional neural network (QCNN).  We follow the training setup as in the paper by having 12 training samples with $\alpha$ equallty distributed in $[0,2]$ and test the models on 500 equally distributed $\alpha$'s.  The only significant difference with the [paper](https://doi.org/10.1103/PRXQuantum.5.020328) is that we utilize the BCE loss for training and only tune the threshold parameter $\tau$ after training.  We tune the threshold parameter $\tau$ just as in the [paper](https://doi.org/10.1103/PRXQuantum.5.020328) by taking the mean of the outputs of the two samples with corresponding $\alpha$'s closest to 1.

The results for $N = 9$ is shown below:

|![](/result_plots/EQCNN.png) ![](/result_plots/HEA_QCNN.png)|
|:--:|
|*EQCNN (first) and QCNN (second)*|

## References
[1] Quynh T. Nguyen et al., Theory for Equivariant Quantum Neural Networks, PRX Quantum 5, 020328, 2024.
