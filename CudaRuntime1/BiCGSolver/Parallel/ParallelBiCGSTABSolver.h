#pragma once
#include "../../Utils/LinearSolver_interface.h"
#include "ParallelBiCGSTABSolverKernel.h"

class ParallelBiCGSTABSolver : public ParallelSolver {
private:
	double* d_mR;
	double* d_mRp;
	double* d_mV;
	double* d_mP;
	double* d_mAx;
	double* d_mS;
	double* d_mT;
	bool* h_mFlag;
public:
	ParallelBiCGSTABSolver(CSR& A, double* X, double* B, const uint32 row, const uint32 col, const uint32 len, const uint32 arrayLen, const uint32 iter, const double bound, ClassType type) :
		ParallelSolver(A, X, B, row, col, len, arrayLen, iter, bound, type)	{}
	ParallelBiCGSTABSolver(CSR& A, double* X, double* B, const uint32 iter, const double bound, ClassType type) :
		ParallelSolver(A, X, B, A.Dimension, A.Dimension, A.Dimension, A.ArrayLength, iter, bound, type) { }

	ParallelBiCGSTABSolver(CSR&& A, double* X, double* B, const uint32 row, const uint32 col, const uint32 len, const uint32 arrayLen, const uint32 iter, const double bound, ClassType type) :
		ParallelSolver(std::move(A), X, B, row, col, len, arrayLen, iter, bound, type) {}
	ParallelBiCGSTABSolver(CSR&& A, double* X, double* B, const uint32 iter, const double bound, ClassType type) :
		ParallelSolver(std::move(A), X, B, A.Dimension, A.Dimension, A.Dimension, A.ArrayLength, iter, bound, type) { }
	virtual void MemoryAllocate() override{
		ParallelSolver::MemoryAllocate();

		cudaError error;
		cudaSetDeviceFlags(cudaDeviceMapHost);

		error = cudaHostAlloc((void**)&h_mFlag, sizeof(bool), cudaHostAllocWriteCombined | cudaHostAllocMapped);
		checkCudaError(error, "CudaMalloc h_mFlag");
		*h_mFlag = false;
		error = cudaMalloc((void**)&d_mR, mRowLength * sizeof(double));
		checkCudaError(error, "CudaMalloc d_mR");
		error = cudaMalloc((void**)&d_mRp, mRowLength * sizeof(double));
		checkCudaError(error, "CudaMalloc d_mRp");
		error = cudaMalloc((void**)&d_mV, mRowLength * sizeof(double));
		checkCudaError(error, "CudaMalloc d_mV");
		error = cudaMalloc((void**)&d_mP, mRowLength * sizeof(double));
		checkCudaError(error, "CudaMalloc d_mP");
		error = cudaMalloc((void**)&d_mAx, mRowLength * sizeof(double));
		checkCudaError(error, "CudaMalloc d_mAx");
		error = cudaMalloc((void**)&d_mS, mRowLength * sizeof(double));
		checkCudaError(error, "CudaMalloc d_mS");
		error = cudaMalloc((void**)&d_mT, mRowLength * sizeof(double));
		checkCudaError(error, "CudaMalloc d_mT");

		error = cudaMemset(d_mV, 0, mRowLength * sizeof(double));
		checkCudaError(error, "CudaMemset d_mV to 0");
		error = cudaMemset(d_mP, 0, mRowLength * sizeof(double));
		checkCudaError(error, "CudaMemset d_mP to 0");
	}
	virtual void MemoryDestroy() override {
		cudaError error;

		error = cudaFree(d_mR);
		checkCudaError(error, "CudaFree d_mR");
		error = cudaFree(d_mRp);
		checkCudaError(error, "CudaFree d_mRp");
		error = cudaFree(d_mV);
		checkCudaError(error, "CudaFree d_mV");
		error = cudaFree(d_mP);
		checkCudaError(error, "CudaFree d_mP");
		error = cudaFree(d_mAx);
		checkCudaError(error, "CudaFree d_mAx");
		error = cudaFree(d_mS);
		checkCudaError(error, "CudaFree d_mS");
		error = cudaFree(d_mT);
		checkCudaError(error, "CudaFree d_mT");
		error = cudaFreeHost(h_mFlag);
		checkCudaError(error, "CudaFree h_mFlag");
	}
	virtual void StartSolving() override {
		std::cout << "并行BiCGSTAB求解中:" << std::endl;
		std::cout << std::endl;

		cudaError error;

		error = cudaMemset(d_mX, 0, mRowLength * sizeof(double));
		checkCudaError(error, "CudaInit d_mX to 0");
		error = cudaMemset(d_mP, 0, mRowLength * sizeof(double));
		checkCudaError(error, "CudaInit d_mP to 0");
		error = cudaMemset(d_mAx, 0, mRowLength * sizeof(double));
		checkCudaError(error, "CudaInit d_mAx to 0");

		double* d_rou1, * d_rou0, * d_w, * d_alpha, *d_vrp, *d_ts, *d_tt, *d_rpt, *d_sum;
		error = cudaMalloc((void**)&d_rou1, sizeof(double));
		checkCudaError(error, "CudaMalloc d_rou1");
		error = cudaMalloc((void**)&d_rou0, sizeof(double));
		checkCudaError(error, "CudaMalloc d_rou0");
		error = cudaMalloc((void**)&d_w, sizeof(double));
		checkCudaError(error, "CudaMalloc d_w");
		error = cudaMalloc((void**)&d_alpha, sizeof(double));
		checkCudaError(error, "CudaMalloc d_alpha");
		error = cudaMalloc((void**)&d_vrp, sizeof(double));
		checkCudaError(error, "CudaMalloc d_vrp");
		error = cudaMalloc((void**)&d_ts, sizeof(double));
		checkCudaError(error, "CudaMalloc d_ts");
		error = cudaMalloc((void**)&d_tt, sizeof(double));
		checkCudaError(error, "CudaMalloc d_tt");
		error = cudaMalloc((void**)&d_rpt, sizeof(double));
		checkCudaError(error, "CudaMalloc d_rpt");
		error = cudaMalloc((void**)&d_sum, sizeof(double));
		checkCudaError(error, "CudaMalloc d_sum");

		bool* d_flag;
		error = cudaHostGetDevicePointer((void**)&d_flag, h_mFlag, 0);
		checkCudaError(error, "cudaHostGetDevicePointer");

		dim3 dimBlock0(THREAD_COMPUTE_MATRIX, 1, 1);
		dim3 dimGrid0((mRowLength - 1) / (THREAD_COMPUTE_MATRIX / WARP_SIZE) + 1, 1, 1);
		uint32 sharedMemorySize0 = THREAD_COMPUTE_MATRIX * 2 * sizeof(double);
		
		dim3 dimBlock1(THREAD_COMPUTE_VECTOR, 1, 1);
		dim3 dimGrid1((mRowLength - 1) / (THREAD_COMPUTE_VECTOR * ELEMS_COMPUTE_VECTOR) + 1, 1, 1);

		MatrixMultiplyVec_CSR << <dimGrid0, dimBlock0, sharedMemorySize0 >> > (d_mA_CSR.Adata, d_mX, d_mAx, d_mA_CSR.RowOffset, d_mA_CSR.ColIndex, mRowLength);

		BiCGSTABInit << <dimGrid1, dimBlock1 >> > (d_mB, d_mAx, d_mR, d_mRp, mRowLength);

		error = cudaMemset(d_rou1, 0, sizeof(double));
		checkCudaError(error, "CudaMemset d_rou1 to 0");
		DotProduct << <dimGrid1, dimBlock1 >> > (d_mR, d_mRp, d_rou1, mRowLength);

		error = cudaMemset(d_rou0, 1, sizeof(double));
		checkCudaError(error, "CudaMemset d_rou0 to 1");
		error = cudaMemset(d_w, 1, sizeof(double));
		checkCudaError(error, "CudaMemset d_w to 1");
		error = cudaMemset(d_alpha, 1, sizeof(double));
		checkCudaError(error, "CudaMemset d_alpha to 1");

		double tol = 0.000001;
		for (mIter = 0; mIter < mMaxIter; ++mIter) {
			//计算mP
			UpdatemP << <dimGrid1, dimBlock1 >> > (d_mP, d_mR, d_mV, d_rou0, d_rou1, d_w, d_alpha, mRowLength);

			//计算mA * mP
			MatrixMultiplyVec_CSR << <dimGrid0, dimBlock0, sharedMemorySize0 >> > (d_mA_CSR.Adata, d_mP, d_mV, d_mA_CSR.RowOffset, d_mA_CSR.ColIndex, mRowLength);

			//mV * mRp
			error = cudaMemset(d_vrp, 0, sizeof(double));
			checkCudaError(error, "cudaMemset d_vrp to 0"); 
			DotProduct << <dimGrid1, dimBlock1 >> > (d_mV, d_mRp, d_vrp, mRowLength);

			//计算mS
			error = cudaMemset(d_sum, 0, sizeof(double));
			checkCudaError(error, "cudaMemset d_sum to 0");
			UpdatemS << <dimGrid1, dimBlock1 >> > (d_mS, d_mR, d_mV, d_rou1, d_vrp, d_alpha, d_sum, mRowLength, mIter);
			
			//判断条件
 			if (*h_mFlag) {
				break;
			}

			//计算mT
			MatrixMultiplyVec_CSR << <dimGrid0, dimBlock0, sharedMemorySize0 >> > (d_mA_CSR.Adata, d_mS, d_mT, d_mA_CSR.RowOffset, d_mA_CSR.ColIndex, mRowLength);

			error = cudaMemset(d_ts, 0, sizeof(double));
			checkCudaError(error, "cudaMemset d_ts to 0");
			error = cudaMemset(d_tt, 0, sizeof(double));
			checkCudaError(error, "cudaMemset d_tt to 0");
			error = cudaMemset(d_rpt, 0, sizeof(double));
			checkCudaError(error, "cudaMemset d_rpt to 0");
			DotProduct << <dimGrid1, dimBlock1 >> > (d_mT, d_mS, d_ts, mRowLength);
			DotProduct << <dimGrid1, dimBlock1 >> > (d_mT, d_mT, d_tt, mRowLength);
			DotProduct << <dimGrid1, dimBlock1 >> > (d_mT, d_mRp, d_rpt, mRowLength);

			//更新mX mR
			UpdatemXmR << <dimGrid1, dimBlock1 >> > (d_mX, d_mR, d_mP, d_mS, d_mT, d_w, d_ts, d_tt, d_rou0,
									d_rou1, d_rpt, d_alpha, d_sum, d_flag, mRowLength, tol, mIter);

		}
		std::cout << "迭代了" << mIter << "次" << std::endl;
		std::cout << std::endl;

		error = cudaFree(d_rou1);
		checkCudaError(error, "CudaFree d_rou1");
		error = cudaFree(d_rou0);
		checkCudaError(error, "CudaFree d_rou0");
		error = cudaFree(d_w);
		checkCudaError(error, "CudaFree d_w");
		error = cudaFree(d_vrp);
		checkCudaError(error, "CudaFree d_vrp");
		error = cudaFree(d_ts);
		checkCudaError(error, "CudaFree d_ts");
		error = cudaFree(d_tt);
		checkCudaError(error, "CudaFree d_tt");
		error = cudaFree(d_rpt);
		checkCudaError(error, "CudaFree d_rpt");
		error = cudaFree(d_sum);
		checkCudaError(error, "CudaFree d_sum");
	}

};