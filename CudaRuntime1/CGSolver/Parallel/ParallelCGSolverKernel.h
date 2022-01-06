#pragma once
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include "../../Main/TypeDef.h"
#include "../../Utils/Contants.h"
#include "../../Utils/Kernels_utils.h"

////得到初始误差 R = B - Ap
////初始化P = R
//__global__ void GetFirstBias(double* B, double* Ap, double* R, double* P, uint32 RowLength) {
//	uint32 tid = threadIdx.x + blockIdx.x * blockDim.x;
//	if (tid < RowLength) {
//		R[tid] = B[tid] - Ap[tid];
//		P[tid] = R[tid];
//	}
//}

//得到初始误差 R = B - Ap
//初始化P = R
__global__ void GetFirstBias(double* B, double* Ap, double* R, double* P, uint32 RowLength) {
	unsigned offset, dataBlockLength;
	calcDataBlockLength<THREAD_COMPUTE_VECTOR, ELEMS_COMPUTE_VECTOR>(offset, dataBlockLength, RowLength);

	for (uint32 tx = threadIdx.x; tx < dataBlockLength; tx += THREAD_COMPUTE_VECTOR) {
		R[offset + tx] = B[offset + tx] - Ap[offset + tx];
		P[offset + tx] = R[offset + tx];
	}
}

//
////设置 Ap = 0
//__global__ void InitMatrix(double* Ap, uint32 RowLength) {
//	uint32 tid = threadIdx.x + blockIdx.x * blockDim.x;
//	if (tid < RowLength) {
//		Ap[tid] = 0.0;
//	}
//}

////更新X与newr
//__global__ void UpdateXAndNewr(double* X, double* Newr, double* R, double* P, double* Ap, double* rr, double* pAp, uint32 RowLength) {
//	uint32 tid = threadIdx.x + blockIdx.x * blockDim.x;
//	double alpha = *rr / *pAp;
//	if (tid < RowLength) {
//		X[tid] += alpha * P[tid];
//		Newr[tid] = R[tid] - alpha * Ap[tid];
//	}
//}

//更新X与newr
__global__ void UpdateXAndNewr(double* X, double* Newr, double* R, double* P, double* Ap, double* rr, double* pAp, bool* flag, uint32 RowLength, double tol) {
	if ( *rr / RowLength < tol) {
		*flag = true;
		//printf("%u \n", iter);
		//return;
	}
	double alpha = *rr / *pAp;
	uint32 offset, dataBlockLength;
	calcDataBlockLength<THREAD_COMPUTE_VECTOR, ELEMS_COMPUTE_VECTOR>(offset, dataBlockLength, RowLength);

	for (uint32 tx = threadIdx.x; tx < dataBlockLength; tx += THREAD_COMPUTE_VECTOR) {
		X[offset + tx] += alpha * P[offset + tx];
		Newr[offset + tx] = R[offset + tx] - alpha * Ap[offset + tx];
	}

}

////更新P与r
//__global__ void UpdateDir(double* P, double* Newr, double* R, double* newrnewr, double* rr, uint32 RowLength) {
//	uint32 tid = threadIdx.x + blockIdx.x * blockDim.x;
//	double beta = *newrnewr / *rr;
//	if (tid < RowLength) {
//		P[tid] = Newr[tid] + beta * P[tid];
//		R[tid] = Newr[tid];
//	}
//}

//更新P与r
__global__ void UpdateDir(double* P, double* Newr, double* R, double* newrnewr, double* rr, uint32 RowLength) {
	double beta = *newrnewr / *rr;
	uint32 offset, dataBlockLength;
	calcDataBlockLength<THREAD_COMPUTE_VECTOR, ELEMS_COMPUTE_VECTOR>(offset, dataBlockLength, RowLength);

	for (uint32 tx = threadIdx.x; tx < dataBlockLength; tx += THREAD_COMPUTE_VECTOR) {
		P[offset + tx] = Newr[offset + tx] + beta * P[offset + tx];
		R[offset + tx] = Newr[offset + tx];
	}
}

//
//判断最大范数，循环退出条件
__global__ void JudgeNormSquare(double* R, double* val, uint32 RowLength) {
	extern __shared__ double scantile[];
	double localVal = 0.0, scanVal = 0.0;
	uint32 offset, dataBlockLength;
	calcDataBlockLength<THREAD_COMPUTE_VECTOR, ELEMS_COMPUTE_VECTOR>(offset, dataBlockLength, RowLength);
	for (uint32 tx = threadIdx.x; tx < dataBlockLength; tx += THREAD_COMPUTE_VECTOR) {
		localVal += R[offset + tx] * R[offset + tx] / RowLength;
	}

	scanVal = intraWarpScan<THREAD_COMPUTE_VECTOR>(scantile, localVal);
	__syncthreads();
	
	if (threadIdx.x == THREAD_COMPUTE_VECTOR - 1) {
		atomicAdd(val, scanVal);
		//printf("%f ", scanVal);
	}
}

//__global__ void DotProduct(double* R1, double* R2, double* rr, bool* flag, uint32 RowLength, double tol, uint32 iter) {
//	uint32 offset, dataBlockLength;
//	double scan = 0.0;
//	double value = 0.0;
//	calcDataBlockLength<THREAD_COMPUTE_VECTOR, ELEMS_COMPUTE_VECTOR>(offset, dataBlockLength, RowLength);
//	for (uint32 tx = threadIdx.x; tx < dataBlockLength; tx += THREAD_COMPUTE_VECTOR) {
//		value += R1[offset + tx] * R2[offset + tx];
//	}
//	scan = intraBlockScan<THREAD_COMPUTE_VECTOR>(value);
//	__syncthreads();
//
//	if (threadIdx.x == THREAD_COMPUTE_VECTOR - 1) {
//		atomicAdd(rr, scan);
//	}
//	__syncthreads();
//
//	if (threadIdx.x == 0 && *rr / RowLength < tol) {
//		*flag = true;
//		printf("%u \n", iter);
//	}
//}


