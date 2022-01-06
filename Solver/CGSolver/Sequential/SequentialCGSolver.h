#pragma once
#include "../../Utils/LinearSolver_interface.h"

class SequentialCGSolver : public LinearSolver {
private:
	double* mR;
	double* mAp;
	double* mNewr;
	double* mP;
public:
	SequentialCGSolver(CSR& A, double* X, double* B, const uint32 row, const uint32 col, const uint32 len, const uint32 arrayLen, const uint32 iter, const double bound, ClassType type) :
		LinearSolver(A, X, B, row, col, len, arrayLen, iter, bound, type) { }
	SequentialCGSolver(CSR& A, double* X, double* B, const uint32 iter, const double bound, ClassType type) :
		LinearSolver(A, X, B, A.Dimension, A.Dimension, A.Dimension, A.ArrayLength, iter, bound, type) { }

	SequentialCGSolver(CSR&& A, double* X, double* B, const uint32 row, const uint32 col, const uint32 len, const uint32 arrayLen, const uint32 iter, const double bound, ClassType type) :
		LinearSolver(std::move(A), X, B, row, col, len, arrayLen, iter, bound, type) { }
	SequentialCGSolver(CSR&& A, double* X, double* B, const uint32 iter, const double bound, ClassType type) :
		LinearSolver(std::move(A), X, B, A.Dimension, A.Dimension, A.Dimension, A.ArrayLength, iter, bound, type) { }

	virtual void MemoryAllocate() override{
#if TIME_TEST
		InstrumentationTimer timer("SequentialCGSolver::MemoryAllocate()");
#endif // 
		LinearSolver::MemoryAllocate();
		mR = new double[mRowLength];
		mAp = new double[mRowLength];
		mNewr = new double[mRowLength];
		mP = new double[mRowLength];
	}
	~SequentialCGSolver() {
		MemoryDestroy();
	}
	virtual void MemoryDestroy() override{
#if TIME_TEST
		InstrumentationTimer timer("SequentialCGSolver::MemoryDestroy()");
#endif // 
		delete[] mR;
		mR = nullptr;

		delete[] mAp;
		mAp = nullptr;

		delete[] mNewr;
		mNewr = nullptr;

		delete[] mP;
		mP = nullptr;
	}

	virtual void StartSolving() override {

#if TIME_TEST
		InstrumentationTimer timer("SequentialCGSolver::StartSolving()");
#endif // 

		//初始化阶段
		for (int i = 0; i < mRowLength; ++i) {
			mX[i] = mNewr[i] = mAp[i] = mR[i] = 0.0;
		}
		//计算Ax 
		Matrix_multi_Vector(mRowLength, mAp, mA_CSR.Adata, mA_CSR.ColIndex, mA_CSR.RowOffset, mX);
		// r = b - Ap  initial p = r
		for (uint32 i = 0; i < mRowLength; ++i) {
			mP[i] =  mR[i] = mB[i] - mAp[i];
		}

		double pAp = 0.0;
		double rr = 0.0;
		double alpha = 0.0;
		double newrnewr = 0.0;
		double beta = 0.0;
		uint32 iter = 0;
		double tol = 0.000001;
		for (; iter < mMaxIter; ++iter) {
			for (uint32 i = 0; i < mRowLength; ++i)
				mAp[i] = 0.0;
			// calculate Ap
			Matrix_multi_Vector(mRowLength, mAp, mA_CSR.Adata, mA_CSR.ColIndex, mA_CSR.RowOffset, mP);
			//for (int i = 0; i < nrows; ++i) {
			//	std::cout << Ap[i] << std::endl;
			//}
			// (p,Ap)
			pAp = dotproduct(mRowLength, mP, mAp);
			//std::cout << "pAp: " << pAp << std::endl;
			// (r,r)
			rr = dotproduct(mRowLength, mR, mR);
			//std::cout << "rr: " << rr << std::endl;

			alpha = rr / pAp;

			if (!(maxnorm(mRowLength, mR) > tol))
				break;

			for (uint32 i = 0; i < mRowLength; ++i) {
				// x = x + alpha * p
				mX[i] += (alpha * mP[i]);
				//std::cout << x[i] << std::endl;
				// r = r - alpha * Ap
				mNewr[i] = mR[i] - alpha * mAp[i];
				//std::cout << newr[i] << std::endl;
			}
			// (r,r)	
			newrnewr = dotproduct(mRowLength, mNewr, mNewr);
			//std::cout << "newrnewr: " << newrnewr << std::endl;
			beta = newrnewr / rr;
			for (uint32 i = 0; i < mRowLength; ++i) {
				// p = r + beta * p
				mP[i] = mNewr[i] + beta * mP[i];
				//std::cout << p[i] << std::endl;
				mR[i] = mNewr[i];
				//std::cout << r[i] << std::endl;
			}

		}
		std::cout << "迭代次数:" << iter << std::endl;
	}
};