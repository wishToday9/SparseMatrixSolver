#pragma once

#include "../../Utils/LinearSolver_interface.h"
#include "../../Utils/BasicOperation_GPU.h"
#include "ParallelCGSolverKernel.h"

class ParallelCGSolver :public ParallelSolver {
private:
	double* d_mR;
	double* d_mAp;
	double* d_mNewr;
	double* d_mP;
	bool* h_mFlag;
public:
	ParallelCGSolver(CSR& A, double* X, double* B, const uint32 row, const uint32 col, const uint32 len, const uint32 arrayLen, const uint32 iter, const double bound, ClassType type) :
		ParallelSolver(A, X, B, row, col, len, arrayLen, iter, bound, type)	{}
	ParallelCGSolver(CSR& A, double* X, double* B, const uint32 iter, const double bound, ClassType type) : 
		ParallelSolver(A, X, B, A.Dimension, A.Dimension, A.Dimension, A.ArrayLength, iter, bound, type) {
	}

	ParallelCGSolver(CSR&& A, double* X, double* B, const uint32 row, const uint32 col, const uint32 len, const uint32 arrayLen, const uint32 iter, const double bound, ClassType type) :
		ParallelSolver(std::move(A), X, B, row, col, len, arrayLen, iter, bound, type) {}
	ParallelCGSolver(CSR&& A, double* X, double* B, const uint32 iter, const double bound, ClassType type) :
		ParallelSolver(std::move(A), X, B, A.Dimension, A.Dimension, A.Dimension, A.ArrayLength, iter, bound, type) {
	}
	virtual void MemoryAllocate() override {
		ParallelSolver::MemoryAllocate();

		cudaError error;
		//启用零拷贝
		cudaSetDeviceFlags(cudaDeviceMapHost);

		error = cudaHostAlloc((void**)&h_mFlag, sizeof(bool), cudaHostAllocWriteCombined | cudaHostAllocMapped);
		checkCudaError(error, "CudaMalloc h_mFlag");
		*h_mFlag = false;

		error = cudaMalloc((void**)&d_mR, mRowLength * sizeof(double));
		checkCudaError(error, "CudaMalloc d_mR");

		error = cudaMalloc((void**)&d_mP, mRowLength * sizeof(double));
		checkCudaError(error, "CudaMalloc d_mP");

		error = cudaMalloc((void**)&d_mAp, mRowLength * sizeof(double));
		checkCudaError(error, "CudaMalloc d_mAp");

		error = cudaMalloc((void**)&d_mNewr, mRowLength * sizeof(double));
		checkCudaError(error, "CudaMalloc d_mNer");

		error = cudaMemset(d_mR, 0, mRowLength * sizeof(double));
		checkCudaError(error, "Cuda d_mR Init Failed");

		error = cudaMemset(d_mAp, 0, mRowLength * sizeof(double));
		checkCudaError(error, "Cuda d_mAp Init Failed");

		error = cudaMemset(d_mNewr, 0, mRowLength * sizeof(double));
		checkCudaError(error, "Cuda d_mNewr Init Failed");

	}
	virtual void MemoryDestroy() override {
		cudaError error;
		error = cudaFreeHost(h_mFlag);
		checkCudaError(error, "CudaFreeHost h_mFlag");
		
		error = cudaFree(d_mR);
		checkCudaError(error, "CudaFree d_mR");

		error = cudaFree(d_mP);
		checkCudaError(error, "CudaFree d_mP");

		error = cudaFree(d_mAp);
		checkCudaError(error, "CudaFree d_mAp");

		error = cudaFree(d_mNewr);
		checkCudaError(error, "CudaFree d_mNewr");
	}

	virtual void StartSolving() override {
		std::cout << "进入并行CG求解" << std::endl;
		cudaError error;
		dim3 dimBlock0(THREAD_COMPUTE_MATRIX, 1, 1);
		dim3 dimGrid0((mRowLength - 1) / (THREAD_COMPUTE_MATRIX / WARP_SIZE) + 1, 1, 1);

		dim3 dimBlock1(THREAD_COMPUTE_VECTOR, 1, 1);
		dim3 dimGrid1((mRowLength - 1) / (THREAD_COMPUTE_VECTOR * ELEMS_COMPUTE_VECTOR) + 1, 1, 1);

		uint32 sharedMemorySize0 = THREAD_COMPUTE_MATRIX * 2 * sizeof(double);
		uint32 sharedMemorySize1 = THREAD_COMPUTE_VECTOR * 2 * sizeof(double);
		//double* data = new double[mRowLength];
		error = cudaMemset(d_mX, 0, mRowLength * sizeof(double));
		checkCudaError(error, "Cuda d_mAp set to 0 failed");
		//InitMatrix << <dimGrid1, dimBlock1 >> > (d_mX, mRowLength);
		//error = cudaMemcpy(data, d_mX, mRowLength * sizeof(double), cudaMemcpyDeviceToHost);
		//for (int i = 0; i < mRowLength; ++i) {
		//	std::cout << data[i] << "    ";
		//}
		MatrixMultiplyVec_CSR << <dimGrid0, dimBlock0, sharedMemorySize0 >> > (d_mA_CSR.Adata, d_mX, d_mAp, d_mA_CSR.RowOffset, d_mA_CSR.ColIndex, mRowLength);

		GetFirstBias << <dimGrid1, dimBlock1 >> > (d_mB, d_mAp, d_mR, d_mP, mRowLength);

		double* d_pAp;
		double* d_rr;
		double* d_newrnewr;
		double* d_beta;
		bool* d_flag;
		error = cudaHostGetDevicePointer((void**)&d_flag, h_mFlag, 0);
		checkCudaError(error, "cudaHostGetDevicePointer");

		error = cudaMalloc((void**)&d_pAp, sizeof(double));
		checkCudaError(error, "CudaMalloc d_pAp");

		error = cudaMalloc((void**)&d_rr, sizeof(double));
		checkCudaError(error, "CudaMalloc d_rr");

		error = cudaMalloc((void**)&d_newrnewr, sizeof(double));
		checkCudaError(error, "CudaMalloc d_newrnewr");

		error = cudaMalloc((void**)&d_beta, sizeof(double));
		checkCudaError(error, "CudaMalloc d_beta");

		double tol = 0.00000000001;
		uint32 iter = 0;
		for (; iter < mMaxIter; ++iter) {
			//TimerClock tc;
			//tc.update();
			//Ap重新置成0;
			error = cudaMemset(d_mAp, 0, mRowLength * sizeof(double));
			checkCudaError(error, "Cuda d_mAp set to 0 failed");

			//计算A*P 放置到AP中
			MatrixMultiplyVec_CSR << <dimGrid0, dimBlock0, sharedMemorySize0 >> > (d_mA_CSR.Adata, d_mP, d_mAp, d_mA_CSR.RowOffset, d_mA_CSR.ColIndex, mRowLength);
			//error = cudaMemcpy(data, d_mAp, mRowLength * sizeof(double), cudaMemcpyDeviceToHost);
			//checkCudaError(error, "error Ap Copy");
			//for (int i = 0; i < mRowLength; ++i) {
			//	std::cout << data[i] << std::endl;
			//}
			//计算pAp
			error = cudaMemset(d_pAp, 0, sizeof(double));
			checkCudaError(error, "set d_pAp = 0");

			DotProduct << <dimGrid1, dimBlock1 >> > (d_mAp, d_mP, d_pAp, mRowLength);
			//error = cudaMemcpy(singleData, d_pAp, sizeof(double), cudaMemcpyDeviceToHost);
			//checkCudaError(error, "error d_pAp single data copy");
			//std::cout << "pAp:" << *singleData << std::endl;

			//计算RR
			error = cudaMemset(d_rr, 0, sizeof(double));
			checkCudaError(error, "set d_rr = 0");
			DotProduct << <dimGrid1, dimBlock1 >> > (d_mR, d_mR, d_rr, mRowLength);
			//JudgeFlag << <1, 1 >> > (d_rr, d_flag, tol, mRowLength);
			//cudaDeviceSynchronize();

			//error = cudaMemcpy(singleData, d_rr, sizeof(double), cudaMemcpyDeviceToHost);
			//checkCudaError(error, "error d_rr single data copy");
			//std::cout << "rr:" << *singleData << std::endl;
			
			//更新x与newr
			UpdateXAndNewr << <dimGrid1, dimBlock1 >> > (d_mX, d_mNewr, d_mR, d_mP, d_mAp, d_rr, d_pAp, d_flag, mRowLength, tol);
			//error = cudaMemcpy(data, d_mX, mRowLength * sizeof(double), cudaMemcpyDeviceToHost);
			//checkCudaError(error, "error d_mX Copy");
			//std::cout << "x: ";
			//for (int i = 0; i < 10; ++i) {
			//	std::cout << data[i] << "    ";
			//}
			//std::cout << std::endl;

			//error = cudaMemcpy(data, d_mNewr, mRowLength * sizeof(double), cudaMemcpyDeviceToHost);
			//checkCudaError(error, "error d_mNewr Copy");
			//for (int i = 0; i < mRowLength; ++i) {
			//	std::cout << data[i] << std::endl;
			//}
			//cudaDeviceSynchronize();
			//计算最大范数
			if (iter > mRowLength && *h_mFlag == true) {
				break;
			}

			//计算newr * newr
			error = cudaMemset(d_newrnewr, 0, sizeof(double));
			checkCudaError(error, "set d_newrnewr = 0");
			DotProduct << <dimGrid1, dimBlock1 >> > (d_mNewr, d_mNewr, d_newrnewr, mRowLength);
			//error = cudaMemcpy(singleData, d_newrnewr, sizeof(double), cudaMemcpyDeviceToHost);
			//checkCudaError(error, "error d_newrnewr single data copy");
			//std::cout << "newrnewr:" << *singleData << std::endl;
			UpdateDir << <dimGrid1, dimBlock1 >> > (d_mP, d_mNewr, d_mR, d_newrnewr, d_rr, mRowLength);
			//error = cudaMemcpy(data, d_mP, mRowLength * sizeof(double), cudaMemcpyDeviceToHost);
			//checkCudaError(error, "error d_mR Copy");
			//std::cout << "p: " ;
			//for (int i = 0; i < 10; ++i) {
			//	std::cout << data[i] << "    ";
			//}
			//std::cout << std::endl;
			//std::cout << tc.getMilliSecond() << " ms" << std::endl;
		}
		std::cout << "迭代次数： " << iter << std::endl;

		error = cudaFree(d_pAp);
		checkCudaError(error, "CudaFree d_pAp");
		
		error = cudaFree(d_rr);
		checkCudaError(error, "CudaFree d_rr");

		error = cudaFree(d_newrnewr);
		checkCudaError(error, "CudaFree d_newrnewr");

		error = cudaFree(d_beta);
		checkCudaError(error, "CudaFree d_beta");

	}


};