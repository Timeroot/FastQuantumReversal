import cirq
import numpy
import scipy.linalg

class PauliSumExponentialGeneral(cirq.Gate):
    def __init__(self, H, t, N):
    	self._N = N
    	self._H = H
    	self._t = t
    	self._U = scipy.linalg.expm(complex(0, t) * H.matrix())
    	
    @property
    def qubits(self):
        return self._pauli_sum.qubits
    
    def _unitary_(self):
        return self._U
    
    def matrix(self):
    	return self._U
    
    def _circuit_diagram_info_(self, args):
        return 'TimeEvolve['+str(self._H)+', '+str(self._t)+']'
    
    def __str__(self):
    	return self._circuit_diagram_info_([]);
    
    def _has_unitary_(self):
    	return True
    
    def _num_qubits_(self):
    	return self._N

"""
Implements a fast list-reversal operation.
If trotter_step is left at zero, it will compute the (dense) unitary time evolution to represent the operation in Cirq.
If trotter_step is nonzero, it will use that many Trotter steps to discretize the evolution; each step being a PauliString that is close to the identity. This will not be an exact reversal, then.
"""
def FastReverseHamiltonian(q, trotter_step=0):
	N = len(q)
	tN = numpy.pi * numpy.sqrt((N+1)**2 - (N % 2)) / 4
	ak = numpy.pi * numpy.sqrt((N+1)**2 - (N + 1 - numpy.arange(2*N + 2))**2) / (4 * tN)
	
	"""
	H = a_1 Sx^1 + a_{2N+1} Sx^{N} + sum_{k=1}^{N-1} a_{2k+1} Sx^k Sx^{k+1} - sum_{k=1}^N a_{2k} Sz^k
	"""
	if trotter_step == 0:
		#Dense
		H = cirq.PauliSum()
		H += complex(ak[1]) * cirq.X(q[0]) # 1 indexed in paper -> 0 indexing in Cirq
		H += complex(ak[2*N+1]) * cirq.X(q[N-1])
		for k in range(1,N):
			H += complex(ak[2*k+1]) * cirq.X(q[k-1]) * cirq.X(q[k])
		
		for k in range(1,N+1):
			H -= complex(ak[2*k]) * cirq.Z(q[k-1])
		
		return [PauliSumExponentialGeneral(H, tN, N).on(*q)]
	else:
		H_x_term = complex(ak[1]) * cirq.X(q[0])
		H_x_term += complex(ak[2*N+1]) * cirq.X(q[N-1])
		for k in range(1,N):
			H_x_term += complex(ak[2*k+1]) * cirq.X(q[k-1]) * cirq.X(q[k])
		H_z_term = cirq.PauliSum()
		for k in range(1,N+1):
			H_z_term -= complex(ak[2*k]) * cirq.Z(q[k-1])
		H_x_term *= tN / trotter_step
		H_z_term *= tN / trotter_step
		res = []
		for step in range(trotter_step):
			#yield cirq.PauliSumExponential(H_x_term)
			#yield cirq.PauliSumExponential(H_z_term)
			res += [cirq.PauliSumExponential(H_x_term)]
			res += [cirq.PauliSumExponential(H_z_term)]
		return res

"""
Similar idea as above, but instead of a time-independent Hamiltonian, it applies (commuting, Pauli) gates. It is twice deep as a normal "swap" network, in terms of gate count, but the gates are all substantially closer to the identity.

If use_pauli_exponential=True, it will use Cirq's native "PauliSumExponential" to process the exponentiated Pauli-X Hamiltonian. If False, it will decompose exactly into two Moments of 2-local gates.

Using conj=True will put the Z operation at the start instead of the end. It takes as long, but may make some simplification more effective.
"""
def FastReverseGates(q, use_pauli_exponential=False, conj=False):
	N = len(q)
	Z_gate = cirq.ZPowGate(exponent = 1 / 2) #S gate
	op_Z = cirq.Moment(map(Z_gate, q))
	
	"""
	* Apply Z_gate on each q
	* Apply X_gate on ends, and XX_gate on middle
	* Repeat steps above N+1 times
	"""
	if use_pauli_exponential:
		X_sum = cirq.X(q[0]) + sum(cirq.X(q[i]) * cirq.X(q[i+1]) for i in range(N-1)) + cirq.X(q[-1])
		op_X = cirq.PauliSumExponential(X_sum, exponent = numpy.pi / 4)
		cycle = [op_X]
	else:
		X_gate = cirq.XPowGate(exponent = 1 / 2) #Often called the SX gate
		XX_gate = cirq.XXPowGate(exponent = 1 / 2)
		if N%2 == 1:
			op_X1 = cirq.Moment([X_gate(q[0])] + list(XX_gate(q[2*i+1], q[2*i+2]) for i in range(N//2)))
			op_X2 = cirq.Moment(list(XX_gate(q[2*i], q[2*i+1]) for i in range(N//2)) + [X_gate(q[-1])])
		else:
			op_X1 = cirq.Moment([X_gate(q[0])] + list(XX_gate(q[2*i+1], q[2*i+2]) for i in range(N//2 - 1)) + [X_gate(q[-1])])
			op_X2 = cirq.Moment(list(XX_gate(q[2*i], q[2*i+1]) for i in range(N//2)))
		cycle = [op_X1, op_X2]
	
	if conj:
		cycle = [op_Z] + cycle
	else:
		cycle = cycle + [op_Z]
	
	res = []
	for s in range(N+1):
		res += cycle
	return res

