#This code is for infidelity simulation of cluster state with n qubtis.
#The method is classical shadow using n qubit clifford.
#The output is # of samplings vs infidelity.
#The infidelity w.r.t. # of samplings is obtained by using bootstrap method.
#Thanks to qiskit, random unitary function is already prepared in the qiskit library.

import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford, Operator


# ---------------- Utility functions ----------------

def random_clifford_numpy(n_qubits, seed=None):
    """Draw an n-qubit random Clifford and return its NumPy matrix."""
    C = random_clifford(n_qubits, seed=seed)
    return Operator(C).data

def fidelity(rho, sigma):
    """Uhlmann fidelity between density matrices."""
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

def reconstruct_from_samples(U_list, outcomes, d):
    """Global-Clifford classical-shadow estimator."""
    avg = np.zeros((d, d), dtype=complex)
    for U, b in zip(U_list, outcomes):
        P = projector_on_outcome(b, d)
        avg += U.conj().T @ P @ U
    avg /= len(U_list)
    return (d + 1) * avg - np.eye(d, dtype=complex)


# ---------------- Cluster-state target ----------------

def cluster_state_density(n_qubits):
    """Return the density matrix of 1D cluster state."""
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.h(i)
    for i in range(n_qubits - 1):
        qc.cz(i, i + 1)
    psi = Operator(qc).data @ np.array([1] + [0]*(2**n_qubits - 1))  # |0...0> -> cluster
    rho = np.outer(psi, psi.conj())
    return rho


# ---------------- Main experiment ----------------

def cluster_shadow_fidelity_numpy(n_qubits, sample_list, seed=None, plot=True):
    """Compute fidelity vs sampling for n-qubit cluster state (NumPy version)."""
    if seed is not None:
        np.random.seed(seed)

    d = 2 ** n_qubits
    rho_true = cluster_state_density(n_qubits)
    fidelities = []

    for Ns in sample_list:
        U_list, outcomes = [], []

        for _ in range(Ns):
            U = random_clifford_numpy(n_qubits)
            rho_rot = U @ rho_true @ U.conj().T
            b = measure_comp_sample(rho_rot)
            U_list.append(U)
            outcomes.append(b)

        rho_hat = reconstruct_from_samples(U_list, outcomes, d)
        F = fidelity(rho_hat, rho_true)
        fidelities.append(F)
        print(f"n={n_qubits}, Ns={Ns:6d} | Fidelity = {F:.4f}")

    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(sample_list, fidelities, marker="o")
        plt.xscale("log")
        plt.xlabel("# of samples (Ns)")
        plt.ylabel(f"Fidelity with cluster_{n_qubits}")
        plt.title(f"Classical-shadow fidelity (NumPy, n={n_qubits})")
        plt.grid(True)
        plt.show()

    return np.array(fidelities)

