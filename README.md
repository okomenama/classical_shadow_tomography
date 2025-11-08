# Classical shadow tomography
Team: QFF foil

Quantum Fall Festival 2025.

# Abstract
Classical Shadow Tomography (CST) is a modern quantum algorithm designed to predict numerous physical quantities, or observables, of a quantum state using significantly fewer measurements than traditional methods. While Quantum State Tomography (QST) attempts to reconstruct the entire density matrix ($\rho$) of the quantum state and requires an exponential number of measurements, CST avoids the comprehensive reconstruction by in prior limiting calculated physical quantitie and only requires polynomial number of measurements, which leads to an efficient estimation of the physical quantities. This classical representation can be post-processed at any time to retrieve the values of various observables that meet specific conditions.

We demonstrated that CST achives efficient estimate of physical quantities of a quantum state through two intrigueing applications. First, we applied CST and QST to GHZ states to estimate expectations of any K-local Pauli observable and show that CST achives more acurate estimate of the local observable expectation with small number of shots than QST. We also show that QST estimate variance does not depend on the number of qubits and only depnds on locality of observable. Additionally, we made a stabilizer circuit simulation program that enables us to apply the CST with the random Clifford unitary ensemble to any stabilizer states, like GHZ states. This circuit is very fast to evaluate the performance for stabilizer states. 
Secondly, we applied the CST to the ground state of the transverse-field Ising model. Since the CST with random Clifford unitaries enables us to store information about stabilizer states using classical memory, we can represent the ground state of the transverse-field Ising model as a classical shadow when the external field strength is weak. This demonstration suggests a potential application of CST to efficiently store the ground state of certain Hamiltonians, provided that an appropriate unitary ensemble is available.

# Code Sescription

# Member
@koizumiyuki
@makkuroym-ops
@TakeruUTSUMI
@wu-hao11
@okomenama
@ransan102

# Mentor
@Keisuke Murota
