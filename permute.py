from reversal import *

import cirq
import numpy
import scipy.linalg

"""
Execute the permutation 'perm' on the qubits 'q' using the reversal operation given.
Defaults to FastReverseGates.

Based on Algorithm 4.1 of Childs.
"""
def FastPermDivideConquer(q, perm, reversal=FastReverseGates, verbose=False):
	N = len(perm)
	if type(perm) != numpy.ndarray:
		perm = numpy.array(perm)
	
	if N == 1:
		return []
	if all(perm == range(N)):
		return []
	if N == 2:
		if verbose:
			print("Swap ",q[0]," <-> ",q[1])
		return [cirq.SWAP(q[0], q[1])]
	
	res = []
	ind01 = 1 * (numpy.array(perm) >= N//2)
	res += sortBinary(q, ind01, perm, reversal, verbose)
	res += FastPermDivideConquer(q[:N//2], perm[:N//2], reversal, verbose)
	res += FastPermDivideConquer(q[N//2:], perm[N//2:] - N//2, reversal, verbose)
	print("Final perm: ",perm)
	return res

"""Subroutine 4.2"""
def sortBinary(q, ind01, perm, reversal, verbose):
	print("Called: ",q,ind01,perm)
	N = len(perm)
	if N == 1:
		return []
	if all(ind01 == sorted(ind01)):
		return []
	if ind01 == [1,0]:
		perm[::-1] = perm
		if verbose:
			print("Swap ",q[0]," <-> ",q[1])
		return [cirq.SWAP(q[0], q[1])]
	
	m1 = N//3
	m2 = (2*N)//3
	res = []
	res += sortBinary(q[0:m1], ind01[0:m1], perm[0:m1], reversal, verbose)
	res += sortBinary(q[m1:m2], 1 - ind01[m1:m2], perm[m1:m2], reversal, verbose)
	res += sortBinary(q[m2:], ind01[m2:], perm[m2:], reversal, verbose)
	
	i = numpy.min(numpy.where(ind01 == 1))
	j = numpy.max(numpy.where(ind01 == 0))
	if verbose:
		print("Reverse ",q[i]," <-> ",q[j])
	res += reversal(q[i:j+1])
	perm[i:j+1] = perm[i:j+1][::-1]
	ind01[i:j+1] = ind01[i:j+1][::-1]
	return res

_=FastPermDivideConquer(q, [0,1,2,4,3,6,5,7], verbose=True);
