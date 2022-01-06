#pragma once
#include <string>
#include <iostream>
#include "../Main/TypeDef.h"
#include "TimeClock.h"
#include "BenchMark.h"
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <iomanip>
#include "BasicOperation_GPU.h"
#include "BasicOperation_CPU.h"
#include "Contants.h"

//解线性方程AX = B
//串行计算基类
class LinearSolver {
protected:
	CSR mA_CSR;  //矩阵存储格式以CSR格式存储
	COO mA_COO;  //矩阵存储格式以COO格式存储
	double* mX;	//向量
	double* mB;	//向量
	const uint32 mRowLength;	//矩阵行数
	const uint32 mColLength;	//矩阵列数
	const uint32 mVecLength;	//向量长度
	const uint32 mNotZerosNum; //不是0的数据长度
	const uint32 mMaxIter;    //最大迭代次数      
	std::string mClassName;	//类名字
	TimerClock mTC;	 //计时
	bool mIsCorrect; //判断结果是否正确
	const double mBound;  //计算误差的界限
	double mDeviation;  //误差
	uint32 mIter;    //迭代次数
	StoreType mStype;  //数据存储类型

public:
	LinearSolver(CSR& A, double* X, double* B, const uint32 row, const uint32 col, const uint32 len, const uint32 arrayLen, const int iter, const double bound, ClassType type) :
		mA_CSR(A), mX(X), mB(B), mRowLength(row), mColLength(col), mVecLength(len), mNotZerosNum(arrayLen), mIsCorrect(false), 
		mMaxIter(iter), mBound(bound), mDeviation(0), mStype(CSRTYPE), mIter {0}, mClassName(className[type])
	{}

	LinearSolver(CSR&& A, double* X, double* B, const uint32 row, const uint32 col, const uint32 len, const uint32 arrayLen, const int iter, const double bound, ClassType type) :
		mA_CSR(std::move(A)), mX(X), mB(B), mRowLength(row), mColLength(col), mVecLength(len), mNotZerosNum(arrayLen), mIsCorrect(false),
		mMaxIter(iter), mBound(bound), mDeviation(0), mStype(CSRTYPE), mIter{ 0 }, mClassName(className[type])
	{}


	virtual ~LinearSolver() {
		MemoryDestroy();
	}
	CSR GetMatrix() const {
		return mA_CSR;
	}
	double* GetBVector() const {
		return mB;
	}
	double* GetXVector() const {
		return mX;
	}
	std::string GetClassName() const{
		return mClassName;
	} 
	uint32 GetRowLength() const {
		return mRowLength;
	}
	uint32 GetColLength() const {
		return mColLength;
	}
	virtual void StartSolving() {
		std::cout << "LinearSolver类中无法完成计算求解" << std::endl;
	}
	virtual void MemoryAllocate() {
	
	}
	virtual void MemoryDestroy() {

#if TIME_TEST
		InstrumentationTimer timer("LinearSolver::MemoryDestroy()");
#endif
		delete[] mB;
		mB = nullptr;

		delete[] mX;
		mX = nullptr;
	}

	virtual double GetCostTime(){
		return mTC.getSecond();
	}

	virtual bool CheckAccuracy() {
#if TIME_TEST
		InstrumentationTimer timer("CheckAccuracy by CPU");
#endif // TIME_TEST

		double* error = new double[mRowLength];
		memset(error, 0, mRowLength * sizeof(double));
		Matrix_multi_Vector(mRowLength, error, mA_CSR.Adata, mA_CSR.ColIndex, mA_CSR.RowOffset, mX);

		for (int i = 0; i < mRowLength; ++i) {
			mDeviation += mB[i] - error[i];
		}
		//累计计算误差处理
		//...
		if (mDeviation >= mBound) {
			mIsCorrect = false;
		}
		else {
			mIsCorrect = true;
		}
		return mIsCorrect;
	};
	
	virtual void Process() {
		mTC.update();
		MemoryAllocate();
		//开始计算解方程
		StartSolving();

		//for (int i = 0; i < mRowLength; i += 200) {
		//	std::cout << "x[" << i << "] = " << mX[i] << std::endl;
		//}

		std::cout << "Compute time cost(内存分配+求解方程): " << mTC.getSecond() << " s" << std::endl;
		std::cout << std::endl;

		//核验结果是否正确
		if (CheckAccuracy()) {
			std::cout << "Compute Correctly!" << std::endl;
			std::cout << "误差: " << mDeviation << std::endl;
		}
		else {
			std::cout << "Error!Error!" << std::endl;
			std::cout << "误差: " << mDeviation << std::endl;
		}
		std::cout << std::endl;

		//MemoryDestroy();
	}
};


//并行计算的基类
class ParallelSolver : public LinearSolver {
protected:
	CSR d_mA_CSR;
	double* d_mX;	//向量
	double* d_mB;	//向量
public:


	//CSR类型存储基类构造
	ParallelSolver(CSR& A, double* X, double* B, const uint32 row, const uint32 col, const uint32 len, const uint32 arrayLen, const uint32 iter,const double bound, ClassType type) :
		LinearSolver(A, X, B, row, col, len, arrayLen, iter,bound, type) { }
	ParallelSolver(CSR&& A, double* X, double* B, const uint32 row, const uint32 col, const uint32 len, const uint32 arrayLen, const uint32 iter, const double bound, ClassType type) :
		LinearSolver(std::move(A), X, B, row, col, len, arrayLen, iter, bound, type) { }

	~ParallelSolver() {
		MemoryDestroy();
	}

	bool checkCudaError(cudaError err, char* discription) {
		if (err != cudaSuccess) {
			const char* errorStr = NULL;
			errorStr = cudaGetErrorName(err);
			fprintf(stderr, "%s(%d): Cuda %s failed! code=%d(%s)\n",
				__FILE__, __LINE__, discription, err, errorStr);
			CleanMem();
			return false;
		}
		return true;
	}

	virtual void CleanMem() {
		if (d_mA_CSR.Adata) {
			cudaFree(d_mA_CSR.Adata);
			d_mA_CSR.Adata = nullptr;
		}
		if (d_mA_CSR.ColIndex) {
			cudaFree(d_mA_CSR.ColIndex);
			d_mA_CSR.ColIndex = nullptr;
		}
		if (d_mA_CSR.RowOffset) {
			cudaFree(d_mA_CSR.RowOffset);
			d_mA_CSR.RowOffset = nullptr;
		}
		if (d_mB) {
			cudaFree(d_mB);
			d_mB = nullptr;
		}
		if (d_mX) {
			cudaFree(d_mX);
			d_mX = nullptr;
		}
	}

	virtual void MemoryAllocate() override {
		cudaError error;
		error = cudaMalloc((void**)&d_mX, mColLength * sizeof(double));
		checkCudaError(error, "CudaMalloc d_mX");

		error = cudaMalloc((void**)&d_mB, mColLength * sizeof(double));
		checkCudaError(error, "CudaMalloc d_mB");
		if (mStype == CSRTYPE) {
			error = cudaMalloc((void**)&d_mA_CSR.Adata, mNotZerosNum * sizeof(double));
			checkCudaError(error, "CudaMalloc d_mA_CSR.Adata");

			error = cudaMalloc((void**)&d_mA_CSR.RowOffset, (mRowLength + 1) * sizeof(uint32));
			checkCudaError(error, "CudaMalloc d_mA_CSR.RowOffset");

			error = cudaMalloc((void**)&d_mA_CSR.ColIndex, mNotZerosNum * sizeof(uint32));
			checkCudaError(error, "CudaMalloc d_mA_CSR.ColIndex");
		}
		else if(mStype == COOTYPE){
			
		}
	}

	virtual void MemoryCopyC2G() {
		cudaError error;
		//error = cudaMemcpy(d_mX, mX, mColLength * sizeof(double), cudaMemcpyHostToDevice);
		//checkCudaError(error, "CudaMemcpyHostToDevice mX to d_mX");
		
		error = cudaMemcpy(d_mB, mB, mColLength * sizeof(double), cudaMemcpyHostToDevice);
		checkCudaError(error, "CudaMemcpyHostToDevice mB to d_mB");

		if (mStype == CSRTYPE) {
			error = cudaMemcpy(d_mA_CSR.Adata, mA_CSR.Adata, mNotZerosNum * sizeof(double), cudaMemcpyHostToDevice);
			checkCudaError(error, "CudaMemcpyHostToDevice mA_CSR.Adata to d_mA_CSR.Adata");

			error = cudaMemcpy(d_mA_CSR.RowOffset, mA_CSR.RowOffset, (mRowLength + 1) * sizeof(uint32), cudaMemcpyHostToDevice);
			checkCudaError(error, "CudaMemcpyHostToDevice mA_CSR.RowOffset to d_mA_CSR.RowOffset");

			error = cudaMemcpy(d_mA_CSR.ColIndex, mA_CSR.ColIndex, mNotZerosNum * sizeof(uint32), cudaMemcpyHostToDevice);
			checkCudaError(error, "CudaMemcpyHostToDevice mA_CSR.ColIndex to d_mA_CSR.ColIndex");
		}
		else if (mStype == COOTYPE) {
			
		}

	}

	virtual void MemoryCopyG2C() {
		cudaError error;
		error = cudaMemcpy(mX, d_mX,mColLength * sizeof(double), cudaMemcpyDeviceToHost);
		checkCudaError(error, "cudaMemcpyDeviceToHost d_mX to mX");

		//error = cudaMemcpy(mB, d_mB, mColLength * sizeof(double), cudaMemcpyDeviceToHost);
		//checkCudaError(error, "cudaMemcpyDeviceToHost d_mB to mB");

		//error = cudaMemcpy(mA_CSR.Adata, d_mA_CSR.Adata, mNotZerosNum * sizeof(double), cudaMemcpyDeviceToHost);
		//checkCudaError(error, "cudaMemcpyDeviceToHost d_mA_CSR.Adata to mA_CSR.Adata");

		//error = cudaMemcpy(mA_CSR.RowOffset, d_mA_CSR.RowOffset, (mRowLength + 1) * sizeof(uint32), cudaMemcpyDeviceToHost);
		//checkCudaError(error, "cudaMemcpyDeviceToHost d_mA_CSR.RowOffset to mA_CSR.RowOffset");

		//error = cudaMemcpy(mA_CSR.ColIndex, d_mA_CSR.ColIndex, mNotZerosNum * sizeof(uint32), cudaMemcpyDeviceToHost);
		//checkCudaError(error, "cudaMemcpyDeviceToHost d_mA_CSR.ColIndex to mA_CSR.ColIndex");
	}

	virtual void MemoryDestroy() override {
		cudaError error;
		if (mStype == CSRTYPE) {
			error = cudaFree(d_mA_CSR.Adata);
			checkCudaError(error, "CudaFree d_mA_CSR.Adata");
			d_mA_CSR.Adata = nullptr;

			cudaFree(d_mA_CSR.ColIndex);
			checkCudaError(error, "CudaFree d_mA_CSR.ColIndex");
			d_mA_CSR.ColIndex = nullptr;

			cudaFree(d_mA_CSR.RowOffset);
			checkCudaError(error, "CudaFree d_mA_CSR.RowOffset");
			d_mA_CSR.RowOffset = nullptr;

			cudaFree(d_mB);
			checkCudaError(error, "CudaFree d_mB");
			d_mB = nullptr;

			cudaFree(d_mX);
			checkCudaError(error, "CudaFree d_mX");
			d_mX = nullptr;
		}
		else if(mStype == COOTYPE) {

		}
	}
	virtual bool CheckAccuracy() {

		//累计计算误差处理
		//...
		cudaError error;
		double* sum, *checkB, *errorB;
		error = cudaMalloc((void**)&sum, sizeof(double));
		checkCudaError(error, "CudaMalloc sum");

		error = cudaMemset(sum, 0, sizeof(double));
		checkCudaError(error, "CudaInit sum to 0");

		error = cudaMalloc((void**)&checkB, mRowLength * sizeof(double));
		checkCudaError(error, "CudaMalloc checkB");

		error = cudaMalloc((void**)&errorB, mRowLength * sizeof(double));
		checkCudaError(error, "CudaMalloc errorB");
		dim3 dimBlock(THREAD_COMPUTE_MATRIX, 1, 1);
		dim3 dimGrid((mRowLength - 1) / (THREAD_COMPUTE_MATRIX / WARP_SIZE) + 1, 1, 1);
		//共享内存大小
		uint32 sharedMemorySize0 = THREAD_COMPUTE_MATRIX * 2 * sizeof(double);

		//error = cudaMemset(d_mX, 0, mRowLength * sizeof(double));
	
		//利用求解出来Ax计算出新的B'
		MatrixMultiplyVec_CSR<< <dimGrid, dimBlock, sharedMemorySize0 >> >(d_mA_CSR.Adata, d_mX, checkB, d_mA_CSR.RowOffset, d_mA_CSR.ColIndex, mRowLength);


		//计算B和B’之间的误差
		dim3 dimB(THREAD_COMPUTE_VECTOR, 1 , 1);
		dim3 dimG((mRowLength - 1) / (THREAD_COMPUTE_VECTOR * ELEMS_COMPUTE_VECTOR) + 1, 1, 1);
		CheckError << <dimG, dimB >> > (d_mB, checkB, errorB, sum, mRowLength);


		error = cudaMemcpy(&mDeviation, sum, sizeof(double), cudaMemcpyDeviceToHost);
		//checkCudaError(error, "CudaMemcpy sum");
		//for (uint32 i = 0; i < mRowLength; ++i) {
		//	//std::cout << std::setprecision(20) << data[i] << std::endl;
		//	printf("%.12f \n", data1[i]);
		//}

		if (mDeviation >= mBound) {
			mIsCorrect = false;
		}
		else {
			mIsCorrect = true;
		}
		return mIsCorrect;
	};

	virtual void Process() override {
		mTC.update();
		MemoryAllocate();
		//拷贝内存到GPU
		MemoryCopyC2G();

		//开始计算解方程
		StartSolving();

		//dim3 dimBlock(THREAD_COMPUTE, 1, 1);
		//dim3 dimGrid((mRowLength - 1) / (THREAD_COMPUTE / WARP_SIZE) + 1, 1, 1);
		//uint32 sharedMemorySize = THREAD_COMPUTE * 2 * sizeof(double);
		//MatrixMultiplyVec_CSR<<<dimGrid, dimBlock, sharedMemorySize >>>(d_mA_CSR.Adata, d_mX, d_mB, d_mA_CSR.RowOffset, d_mA_CSR.ColIndex, mRowLength);

		//dim3 dimB(THREAD_COMPUTE, 1, 1);
		//dim3 dimG((mRowLength - 1) / THREAD_COMPUTE + 1, 1, 1);

		//double* d_val;
		//cudaMalloc((void**)&d_val, sizeof(double));

		//DotProduct << <dimG, dimB >> > (d_mX, d_mB, d_val, mRowLength);
		//double* h_val = new double;
		//
		//cudaMemcpy(h_val, d_val, sizeof(double), cudaMemcpyDeviceToHost);
		//std::cout << *h_val << std::endl;

		//拷贝数据到CPU
		MemoryCopyG2C();

		for (int i = 0; i < mRowLength; i += 200) {
			std::cout << "x[" << i << "] = " << mX[i] << std::endl;
		}

		std::cout << "Compute time cost: " << mTC.getSecond() << " s" << std::endl;
		std::cout << std::endl;

		//核验结果是否正确
		if (CheckAccuracy()) {
			std::cout << "Compute Correctly!" << std::endl;
			std::cout << "误差: " << mDeviation << std::endl;
		}
		else {
			std::cout << "Error!Error!" << std::endl;
			std::cout << "总误差: " << mDeviation << std::endl;
		}
	}

	virtual void StartSolving() override {
		std::cout << "ParallelSolver类中无法完成计算求解" << std::endl;
	}
};
