#pragma once
#include "../../Utils/LinearSolver_interface.h"
#include "../../Utils/BasicOperation_CPU.h"

class SequtialBiCGSTABSolver : public LinearSolver {
private:
	double* mR;
	double* mRp;
	double* mV;
	double* mP;
	double* mAx;
	double* mS;
	double* mT;
public:
	SequtialBiCGSTABSolver(CSR& A, double* X, double* B, const uint32 row, const uint32 col, const uint32 len, const uint32 arrayLen, const uint32 iter, const double bound, ClassType type) :
		LinearSolver(A, X, B, row, col, len, arrayLen, iter, bound, type){ }
	SequtialBiCGSTABSolver(CSR& A, double* X, double* B,const uint32 iter, const double bound, ClassType type) :
		LinearSolver(A, X, B, A.Dimension, A.Dimension, A.Dimension, A.ArrayLength, iter, bound, type) {	}

	SequtialBiCGSTABSolver(CSR&& A, double* X, double* B, const uint32 row, const uint32 col, const uint32 len, const uint32 arrayLen, const uint32 iter, const double bound, ClassType type) :
		LinearSolver(A, X, B, row, col, len, arrayLen, iter, bound, type) { }
	SequtialBiCGSTABSolver(CSR&& A, double* X, double* B, const uint32 iter, const double bound, ClassType type) :
		LinearSolver(A, X, B, A.Dimension, A.Dimension, A.Dimension, A.ArrayLength, iter, bound, type) {	}

	virtual void MemoryAllocate() override {
		LinearSolver::MemoryAllocate();
		//分配内存
		mR = new double[mRowLength];
		mRp = new double[mRowLength];
		mV = new double[mRowLength];
		mP = new double[mRowLength];
		mAx = new double[mRowLength];
		mS = new double[mRowLength];
		mT = new double[mRowLength];
	}

	virtual void MemoryDestroy() override {
		delete[] mR;
		mR = nullptr;

		delete[] mRp;
		mRp = nullptr;

		delete[] mV;
		mV = nullptr;

		delete[] mP;
		mP = nullptr;

		delete[] mAx;
		mAx = nullptr;

		delete[] mS;
		mS = nullptr;

		delete[] mT;
		mT = nullptr;
	}

	virtual void StartSolving() override {
		//初始化x
		memset(mX, 0, mRowLength * sizeof(double));
		memset(mV, 0, mRowLength * sizeof(double));
		memset(mP, 0, mRowLength * sizeof(double));
		memset(mAx, 0, mRowLength * sizeof(double));

		double* error = new double[mRowLength];
		//计算Ax
		Matrix_multi_Vector(mRowLength, mAx, mA_CSR.Adata, mA_CSR.ColIndex, mA_CSR.RowOffset, mX);
		for (uint32 i = 0; i < mRowLength; ++i) {
			mRp[i] = mR[i] = mB[i] - mAx[i];
		}

		double rou1 = dotproduct(mRowLength, mRp, mR);
		double rou0, w, alpha;
		rou0 = w = alpha = 1;
		uint32 iter = 0;
		double tol = 0.000001;
		for (; iter < mMaxIter; ++iter) {
			//if (squarenorm(mRowLength, mR) < tol * squarenorm(mRowLength, mB)) {
			//	break;
			//}
			double beta = rou1 * alpha / (rou0 * w);
			for (uint32 i = 0; i < mRowLength; ++i) {
				mP[i] = mR[i] + beta * (mP[i] - w * mV[i]);
			}
			//计算Ap
			memset(mV, 0, mRowLength * sizeof(double));
			Matrix_multi_Vector(mRowLength, mV, mA_CSR.Adata, mA_CSR.ColIndex, mA_CSR.RowOffset, mP);

			alpha = rou1 / (dotproduct(mRowLength, mV, mRp));

			for (uint32 i = 0; i < mRowLength; ++i) {
				mS[i] = mR[i] - alpha * mV[i];
			}
			if (squarenorm(mRowLength, mS) < tol) {
				break;
			}
			std::cout << "norm:" << squarenorm(mRowLength, mS) << std::endl;
			//计算As
			memset(mT, 0, mRowLength * sizeof(double));
			Matrix_multi_Vector(mRowLength, mT, mA_CSR.Adata, mA_CSR.ColIndex, mA_CSR.RowOffset, mS);

			w = dotproduct(mRowLength, mT, mS) / dotproduct(mRowLength, mT, mT);

			rou0 = rou1;
			rou1 = -w * dotproduct(mRowLength, mRp, mT);

			for (uint32 i = 0; i < mRowLength; ++i) {
				mX[i] = mX[i] + alpha * mP[i] + w * mS[i];
				mR[i] = mS[i] - w * mT[i];
			}
			//memset(error, 0, mRowLength * sizeof(double));
			//Matrix_multi_Vector(mRowLength, error, mA_CSR.Adata, mA_CSR.ColIndex, mA_CSR.RowOffset, mX);
			//double sum = 0;
			//for (int i = 0; i < mRowLength; ++i) {
			//	sum += mB[i] - error[i];
			//}
			//std::cout << "误差和:" << sum << std::endl;
		}
		std::cout << "迭代次数：" << iter << std::endl;

	}


};