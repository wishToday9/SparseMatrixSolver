#pragma once
#include "Kernels_utils.h"
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include "../Main/TypeDef.h"


////计算向量点乘 并且将结果存放在val中
//__global__ void DotProduct(double* a, double* b, double* val, uint32 len) {
//	uint32 tid = threadIdx.x + blockIdx.x * blockDim.x;
//	double scan = 0;
//	double value = 0;;
//
//	if (tid < len) {
//		value = a[tid] * b[tid];
//	}
//	scan = intraBlockScan<THREAD_COMPUTE>(value);
//	__syncthreads();
//
//
//	if (threadIdx.x == THREAD_COMPUTE - 1) {
//		atomicAdd(val, scan);
//	}
//}
// 
//计算向量点乘 并且将结果存放在val中（一定要把val的初值设成0，否则会与原始值累加倒置计算错误）
__global__ void DotProduct(double* a, double* b, double* val, uint32 len) {
	uint32 offset, dataBlockLength;
	double scan = 0.0;
	double value = 0.0;
	calcDataBlockLength<THREAD_COMPUTE_VECTOR, ELEMS_COMPUTE_VECTOR>(offset, dataBlockLength, len);
	for (uint32 tx = threadIdx.x; tx < dataBlockLength; tx += THREAD_COMPUTE_VECTOR) {
		value += a[offset + tx] * b[offset + tx];
	}
	scan = intraBlockScan<THREAD_COMPUTE_VECTOR>(value);
	__syncthreads();

	if (threadIdx.x == THREAD_COMPUTE_VECTOR - 1) {
		atomicAdd(val, scan);
	}
}

//计算矩阵 * 向量 a * b = c， 将结果存放在c中
__global__ void MatrixMultiplyVec_CSR(double* a, double* b, double* c, uint32* RowOffset, uint32* colIndex, uint32 RowLength) {
	extern __shared__ double scanTile[];
	uint32 tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint32 warpID = tid / WARP_SIZE;
	uint32 lane = threadIdx.x & (WARP_SIZE - 1);
	uint32 offset = blockDim.x / WARP_SIZE * gridDim.x;
	double localVal = 0.0, scanVal = 0.0;
	for (uint32 row = warpID; row < RowLength; row += offset) {
		uint32 localLength = RowOffset[row + 1] - RowOffset[row];
		for (uint32 i = lane; i <= localLength - 1; i += WARP_SIZE) {
			localVal += a[RowOffset[row] + i] * b[colIndex[RowOffset[row] + i]];  //计算
		}
		scanVal = intraWarpScan<THREAD_COMPUTE_MATRIX>(scanTile, localVal) + localVal;
		//printf("%f", localVal);
		if (lane == WARP_SIZE - 1) {
			c[row] = scanVal;
			//printf("%f", scanVal);
		}
	}
}



//求出两个矩阵的误差
__global__ void CheckError(double* B, double* NewB, double* ErrorB, double* val, uint32 len) {
	uint32 offset, dataBlockLength;
	double scan = 0.0;
	double value = 0.0;
	calcDataBlockLength<THREAD_COMPUTE_VECTOR, ELEMS_COMPUTE_VECTOR>(offset, dataBlockLength, len);
	for (uint32 tx = threadIdx.x; tx < dataBlockLength; tx += THREAD_COMPUTE_VECTOR) {
		ErrorB[offset + tx] = B[offset + tx] - NewB[offset + tx];
		value += ErrorB[offset + tx];
	}
	scan = intraBlockScan<THREAD_COMPUTE_VECTOR>(value);
	__syncthreads();

	if (threadIdx.x == THREAD_COMPUTE_VECTOR - 1) {
		atomicAdd(val, scan);
	}
}
