#This code is for infidelity simulation of cluster state with n qubtis.
#The method is classical shadow using tensor products of one qubit clifford.
#The output is # of samplings vs infidelity.
#The infidelity w.r.t. # of samplings is obtained by using bootstrap method.
#Thanks to qiskit, random unitary function is already prepared in the qiskit library. 

import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford, Operator


# ---------------- Utility functions ----------------

def random_local_clifford_numpy(n_qubits):
    """Generate a tensor product of single-qubit random Cliffords as a NumPy unitary."""
    U_total = np.array([[1]], dtype=complex)
    for _ in range(n_qubits):
        C = random_clifford(1)
        U_total = np.kron(U_total, Operator(C).data)
    return U_total

def fidelity(rho, sigma):
    """Uhlmann fidelity between two density matrices."""
    sr = sqrtm(rho)
    inner = sr @ sigma @ sr
    return float(np.real(np.trace(sqrtm(inner))) ** 2)

def measure_comp_sample(rho):
    """Sample a computational-basis outcome from ρ."""
    probs = np.real_if_close(np.diag(rho))
    probs = np.maximum(probs, 0)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)

def projector_on_outcome(b, d):
    P = np.zeros((d, d), dtype=complex)
    P[b, b] = 1.0
    return P

def reconstruct_from_samples_local_cliff(U_list, outcomes, d):
    """Local-Clifford classical-shadow estimator."""
    avg = np.zeros((d, d), dtype=complex)
    for U, b in zip(U_list, outcomes):
        P = projector_on_outcome(b, d)
        avg += U.conj().T @ P @ U
    avg /= len(U_list)
    # Inverse map for d-dim full Hilbert space
    return (d + 1) * avg - np.eye(d, dtype=complex)

# Here we use 1D cluster state as an example for tomography.
# You can update here for any kind of state for tomography.
def cluster_state_density(n_qubits):
    """Return the density matrix of 1D cluster state."""
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.h(i)
    for i in range(n_qubits - 1):
        qc.cz(i, i + 1)
    psi = Operator(qc).data @ np.array([1] + [0]*(2**n_qubits - 1))
    rho = np.outer(psi, psi.conj())
    return rho


# ---------------- Bootstrap experiment ----------------

def cluster_shadow_infidelity_bootstrap_local(
    n_qubits, sample_list, n_bootstrap=100, seed=None, plot=True
):
    """
    Estimate infidelity |F−1| for cluster-state reconstruction using tensor-product (local) Cliffords.
    """

    if seed is not None:
        np.random.seed(seed)

    d = 2 ** n_qubits
    rho_true = cluster_state_density(n_qubits)

    mean_infid, std_infid = [], []

    for Ns in sample_list:
        # --- Base shadow samples ---
        U_list, outcomes = [], []
        for _ in range(Ns):
            U = random_local_clifford_numpy(n_qubits)
            rho_rot = U @ rho_true @ U.conj().T
            b = measure_comp_sample(rho_rot)
            U_list.append(U)
            outcomes.append(b)

        # --- Bootstrap resampling ---
        infids = []
        for _ in range(n_bootstrap):
            idxs = np.random.choice(Ns, Ns, replace=True)
            U_resamp = [U_list[i] for i in idxs]
            b_resamp = [outcomes[i] for i in idxs]
            rho_hat = reconstruct_from_samples_local_cliff(U_resamp, b_resamp, d)
            F = fidelity(rho_hat, rho_true)
            infids.append(abs(F - 1))

        mean_infid.append(np.mean(infids))
        std_infid.append(np.std(infids))
        print(f"Ns={Ns:<6d} | mean(|F-1|)={mean_infid[-1]:.4e} ± {std_infid[-1]:.4e}")

    if plot:
        plt.figure(figsize=(6,4))
        plt.errorbar(sample_list, mean_infid, yerr=std_infid, fmt='o-', capsize=4)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Number of samples (Ns)")
        plt.ylabel("Bootstrap mean |F−1| ± std")
        plt.title(f"Bootstrap infidelity (local Cliffords, n={n_qubits} cluster)")
        plt.grid(True, which='both', ls='--')
        plt.show()

    return np.array(mean_infid), np.array(std_infid)
