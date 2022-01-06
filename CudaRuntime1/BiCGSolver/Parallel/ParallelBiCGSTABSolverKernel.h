#pragma once
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_types.h>
#include "../../Utils/Kernels_utils.h"
#include "../../Main/TypeDef.h"

__global__ void BiCGSTABInit(double* B, double* Ax, double* R, double* Rp, uint32 RowLength) {
	unsigned offset, dataBlockLength;
	calcDataBlockLength<THREAD_COMPUTE_VECTOR, ELEMS_COMPUTE_VECTOR>(offset, dataBlockLength, RowLength);

	for (uint32 tx = threadIdx.x; tx < dataBlockLength; tx += THREAD_COMPUTE_VECTOR) {
		Rp[offset + tx] = R[offset + tx] = B[offset + tx] - Ax[offset + tx];
	}

}

__global__ void UpdatemP(double* P, double* R, double* V, double* rou0, double* rou1, double* w, double* alpha, uint32 RowLength) {
	unsigned offset, dataBlockLength;
	calcDataBlockLength<THREAD_COMPUTE_VECTOR, ELEMS_COMPUTE_VECTOR>(offset, dataBlockLength, RowLength);
	double beta = (*rou1 / *rou0) * (*alpha / *w);

	for (uint32 tx = threadIdx.x; tx < dataBlockLength; tx += THREAD_COMPUTE_VECTOR) {
		P[offset + tx] = R[offset + tx] + beta * (P[offset + tx] - *w * V[offset + tx]);
	}
}

__global__ void UpdatemS(double* S, double* R, double* V, double* rou1, double* vrp, double* alpha, double* sum, uint32 RowLength, uint32 Iter) {
	unsigned offset, dataBlockLength;
	calcDataBlockLength<THREAD_COMPUTE_VECTOR, ELEMS_COMPUTE_VECTOR>(offset, dataBlockLength, RowLength);
	*alpha = (*rou1) / (*vrp);

	double localVal = 0.0;
	double scanVal = 0.0;
	for (uint32 tx = threadIdx.x; tx < dataBlockLength; tx += THREAD_COMPUTE_VECTOR) {
		S[offset + tx] = R[offset + tx] - (*alpha) * V[offset + tx];
		localVal += S[offset + tx] * S[offset + tx];
	}
	scanVal = intraBlockScan<THREAD_COMPUTE_VECTOR>(localVal);
	__syncthreads();

	if (threadIdx.x == THREAD_COMPUTE_VECTOR - 1) {
		atomicAdd(sum, scanVal);
	}

}

__global__ void UpdatemXmR(double* X, double* R, double* P, double* S, double* T, double* w, double* ts, double* tt,
		double* rou0, double* rou1, double* rpt, double* alpha, double* sum, bool* flag, uint32 RowLength, double tol, uint32 Iter) {
	if (sqrt((*sum))  < tol) {
		*flag = true;
	}
	unsigned offset, dataBlockLength;
	calcDataBlockLength<THREAD_COMPUTE_VECTOR, ELEMS_COMPUTE_VECTOR>(offset, dataBlockLength, RowLength);
	*w = (*ts) / (*tt);
	*rou0 = *rou1;
	*rou1 = -(*w) * (*rpt);

	for (uint32 tx = threadIdx.x; tx < dataBlockLength; tx += THREAD_COMPUTE_VECTOR) {
		X[offset + tx] = X[offset + tx] + (*alpha) * P[offset + tx] + (*w) * S[offset + tx];
		R[offset + tx] = S[offset + tx] - (*w) * T[offset + tx];
	}
}

