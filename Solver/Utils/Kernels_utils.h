#pragma once

#include "Contants.h"
#include <cuda_runtime_api.h>

#define max(x,y)	((x)<(y)?(y):(x)) //using the max,min function from <algorithm>
#define min(x,y)	((x)<(y)?(x):(y))

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

#else
__device__ double atomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val +
				__longlong_as_double(assumed)));

		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);

	return __longlong_as_double(old);
}
#endif

template<uint32 blockSize>
inline __device__ double intraWarpScan(volatile double* scanTile, double val) {
	unsigned index = 2 * threadIdx.x - (threadIdx.x & (min(blockSize, WARP_SIZE) - 1));

	scanTile[index] = 0;              // ��ǰ��һ������
	index += min(blockSize, WARP_SIZE);
	scanTile[index] = val;

	if (blockSize >= 2)
	{
		scanTile[index] += scanTile[index - 1];
	}

	if (blockSize >= 4)
	{
		scanTile[index] += scanTile[index - 2];
	}
	if (blockSize >= 8)
	{
		scanTile[index] += scanTile[index - 4];
	}
	if (blockSize >= 16)
	{
		scanTile[index] += scanTile[index - 8];
	}
	if (blockSize >= 32)
	{
		scanTile[index] += scanTile[index - 16];
	}
	// ���Ԫ�ص�ֵ���кϲ�
	return scanTile[index] - val;
}

template<uint32 blockSize>
inline __device__ unsigned intraWarpScan(volatile unsigned* scanTile, unsigned val) {
	unsigned index = 2 * threadIdx.x - (threadIdx.x & (min(blockSize, WARP_SIZE) - 1));

	scanTile[index] = 0;              // ��ǰ��һ������
	index += min(blockSize, WARP_SIZE);
	scanTile[index] = val;

	if (blockSize >= 2)
	{
		scanTile[index] += scanTile[index - 1];
	}

	if (blockSize >= 4)
	{
		scanTile[index] += scanTile[index - 2];
	}
	if (blockSize >= 8)
	{
		scanTile[index] += scanTile[index - 4];
	}
	if (blockSize >= 16)
	{
		scanTile[index] += scanTile[index - 8];
	}
	if (blockSize >= 32)
	{
		scanTile[index] += scanTile[index - 16];
	}
	// ���Ԫ�ص�ֵ���кϲ�
	return scanTile[index] - val;
}

template<uint32 blockSize>
inline __device__ double intraBlockScan(double val) {
	__shared__ double scanTile[blockSize * 2];                 // ����Ҫ�����Ĺ������ж�󣿣���
	unsigned warpIdx = threadIdx.x / WARP_SIZE;
	unsigned laneIdx = threadIdx.x & (WARP_SIZE - 1);  // Thread index inside warp


	double warpResult = intraWarpScan<blockSize>(scanTile, val);
	__syncthreads();


	if (laneIdx == WARP_SIZE - 1)                 // �õ�32��ֵ���ܺͷ��ڶ�Ӧ��warpIdx��
	{
		scanTile[warpIdx] = warpResult + val;
	}
	__syncthreads();


	if (threadIdx.x < WARP_SIZE)                  // ��������һ��warp���в���
	{
		scanTile[threadIdx.x] = intraWarpScan< blockSize / WARP_SIZE>(scanTile, scanTile[threadIdx.x]);
	}
	__syncthreads();

	return warpResult + scanTile[warpIdx] + val;
}


template<uint32 blockSize>
inline __device__ unsigned intraBlockScan(unsigned val) {
	__shared__ unsigned scanTile[blockSize * 2];                 // ����Ҫ�����Ĺ������ж�󣿣���
	unsigned warpIdx = threadIdx.x / WARP_SIZE;
	unsigned laneIdx = threadIdx.x & (WARP_SIZE - 1);  // Thread index inside warp

	unsigned warpResult = intraWarpScan<blockSize>(scanTile, val);
	__syncthreads();


	if (laneIdx == WARP_SIZE - 1)                 // �õ�32��ֵ���ܺͷ��ڶ�Ӧ��warpIdx��
	{
		scanTile[warpIdx] = warpResult + val;
	}
	__syncthreads();


	if (threadIdx.x < WARP_SIZE)                  // ��������һ��warp���в���
	{
		scanTile[threadIdx.x] = intraWarpScan<blockSize / WARP_SIZE>(scanTile, scanTile[threadIdx.x]);
	}
	__syncthreads();

	return warpResult + scanTile[warpIdx] + val;
}

template <unsigned numThreads, unsigned elemsThread>
inline __device__ void calcDataBlockLength(unsigned& offset, unsigned& dataBlockLength, unsigned arrayLength)
{
	unsigned elemsPerThreadBlock = numThreads * elemsThread;            // ����ÿ���߳̿�Ҫ�����������
	offset = blockIdx.x * elemsPerThreadBlock;
	dataBlockLength = offset + elemsPerThreadBlock <= arrayLength ? elemsPerThreadBlock : arrayLength - offset;       // �����һ���߳̿�����⴦��
}

