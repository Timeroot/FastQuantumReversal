from reversal import *

import cirq
import numpy
import scipy.linalg

"""
Do a faster-than-obvious SWAP operation between two qubits, using a reversal method of choice.
Defaults to FastReverseGates.
When the qubits are neighbors, falls back to regular SWAP.

Could be improved by using a custom implementation for three qubits, definitely.
"""
def FastSwap(q, i, j, reversal = FastReverseGates):
	if i == j:
		return []
	if abs(i - j) == 1:
		return [cirq.SWAP(q[i], q[j])]
	if abs(i - j) == 2:
		return [cirq.SWAP(q[i], q[i+1]), cirq.SWAP(q[j], q[i+1]), cirq.SWAP(q[i], q[i+1])]
	rev1 = reversal(q[i:j+1])
	if reversal == FastReverseGates: #use conj to simplify
		rev2 = reversal(q[i+1:j], conj=True)
	else:
		rev2 = reversal(q[i+1:j])
	return rev1 + rev2

