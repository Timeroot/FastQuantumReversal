import cirq
import numpy
import scipy.linalg

from reversal import *
from swap import *

N = 8
q = cirq.LineQubit.range(N) #list of our qubits
obs = list(map(lambda qb: (1-cirq.Z(qb))/2, q)) #list of observables (output of the circuit)
simulator = cirq.Simulator() #simulator we'll use

opt_merge1 = cirq.MergeSingleQubitGates()
opt_ejZ = cirq.EjectZ()
opt_merge2 = cirq.MergeInteractions()
opt_drop = cirq.DropEmptyMoments()
opt = lambda x: (opt_merge1(circ), opt_drop(circ))

#Reverse the seven qubits.
circ = cirq.Circuit(FastReverseHamiltonian(q))
simulator.simulate_expectation_values(circ, observables=obs, initial_state=0b0110001)
#Prints: [(1+0j), 0j, 0j, 0j, (1+0j), (1+0j), 0j]

#Out of the qubits 1-7, reverse just the list 2-4:
circ = cirq.Circuit((cirq.I(qb) for qb in q), FastReverseHamiltonian(q[1:4]))
simulator.simulate_expectation_values(circ, observables=obs, initial_state=0b0110001)
#Prints: [0j, 0j, (1+0j), (1+0j), 0j, 0j, (1+0j)]

#Test the Trotter expansion of the reversal Hamiltonian:
circ = cirq.Circuit(FastReverseHamiltonian(q, 30))
simulator.simulate_expectation_values(circ, observables=obs, initial_state=0b0110001)
#Prints: [(0.9975656867027283+0j), (0.0037863552570343018+0j), (0.005503922700881958+0j), (0.0031128525733947754+0j), (0.9967929422855377+0j), (0.9962047338485718+0j), (0.002434283494949341+0j)]

circ = cirq.Circuit(FastReverseGates(q))
simulator.simulate_expectation_values(circ, observables=obs, initial_state=0b0110001)

circ = cirq.Circuit(FastReverseGates(q, False))
simulator.simulate_expectation_values(circ, observables=obs, initial_state=0b0110001)

#Test some swaps
circ = cirq.Circuit((cirq.I(qb) for qb in q), FastSwap(q, 0, 5))
simulator.simulate_expectation_values(circ, observables=obs, initial_state=0b0110001)
simulator.simulate_expectation_values(circ, observables=obs, initial_state=0b1110001)
simulator.simulate_expectation_values(circ, observables=obs, initial_state=0b0110101)
simulator.simulate_expectation_values(circ, observables=obs, initial_state=0b0110011)
