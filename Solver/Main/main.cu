#include "../BiCGSolver/Parallel/ParallelBiCGSTABSolver.h"
#include "../BiCGSolver/Sequential/SequtialBiCGSTABSolver.h"
#include "../CGSolver/Parallel/ParallelCGSolver.h"
#include "../CGSolver/Sequential/SequentialCGSolver.h"
#include "../Utils/LinearSolver_interface.h"

void Matrix_multi_Vector(int nrows, double* ax, double* sra, uint32* clm, uint32* fnz, double* x) {
	for (int i = 0; i < nrows; ++i)
		for (int j = fnz[i]; j < fnz[i + 1]; ++j)
			ax[i] += (sra[j] * x[clm[j]]);
}
void readB(double *&Bdata ,std::string path) {
	uint32 n, size;
	std::ifstream file2(path); // dhmiourgoume to vector 
	std::string str2;
	getline(file2, str2);
	std::stringstream streamB1(str2);
	streamB1 >> n;
	size = n;
	Bdata = new double[size];
	getline(file2, str2);   // fortonoume tis times sto vector mas   
	std::stringstream streamB2(str2);
	uint32 i = 0;
	double v;
	while (streamB2 >> v) {
		//if (!stream) break;
		Bdata[i] = v;
		//std::cout << Bdata[i] << "   ";
		i++;
	}
	//std::cout << "结束！" << std::endl;
}

int main() {
	double start_t, finish_t;

	int nrows = 5000;
	int iter = 5000;
	//数据
	double* sra = new double[3 * nrows - 2];
	sra[0] = 1.0;
	for (int i = 1; i < 3 * nrows - 2; ++i) {
		if (i % 3 == 1)
			sra[i] = (int)(i / 3) + 2;
		else if (i % 3 == 2)
			sra[i] = (int)(i / 3) + 2;
		else
			sra[i] = (int)(i / 3) + 1;
	}

	//列索引
	uint32* clm = new uint32[3 * nrows - 2];
	clm[0] = 0;
	clm[1] = 1;
	for (int i = 2; i < 3 * nrows - 4; ++i) {
		if (i % 3 == 2)
			clm[i] = (int)(i / 3);
		else if (i % 3 == 0)
			clm[i] = (int)(i / 3);
		else
			clm[i] = (int)(i / 3) + 1;
	}
	clm[3 * nrows - 4] = nrows - 2;
	clm[3 * nrows - 3] = nrows - 1;

	uint32* fnz = new uint32[nrows + 1];
	fnz[0] = 0;
	fnz[1] = 2;
	for (int i = 2; i < nrows; ++i)
		fnz[i] = fnz[i - 1] + 3;
	fnz[nrows] = 3 * nrows - 2;

	/*
	double sra[28] = {1,2, 2,2,3, 3,3,4, 4,4,5, 5,5,6, 6,6,7, 7,7,8, 8,8,9, 9,9,10, 10,10};
	int clm[28] = {0,1, 0,1,2, 1,2,3, 2,3,4, 3,4,5, 4,5,6, 5,6,7, 6,7,8, 7,8,9,  8,9};
	int fnz[11] = {0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 28};
	*/

	double* b = new double[nrows];
	for (int i = 0; i < nrows; ++i)
		b[i] = 0;

	printf("The exact solution of x is:\n");

	//x的解
	double* x = new double[nrows];
	for (int i = 0; i < nrows; ++i) {
		x[i] = 1.0 / (i + 3.14);
		//x[i] = i;
		//printf("x[%d] = %f\n",i,x[i]);
	}

	printf("Solving the equation...\n\n");
	// calculate b
	//算出来b
	Matrix_multi_Vector(nrows, b, sra, clm, fnz, x);

	for (int i = 0; i < nrows; ++i) {
		//std::cout << b[i] << std::endl;
		x[i] = 0.0;
	}

	//CSR A;
	//A.Adata = sra;
	//A.ColIndex = clm;
	//A.RowOffset = fnz;

	double* newB;
	CSR B("./A.txt");
	readB(newB, "./B.txt");
	//ParallelSolver solver1(A, x, b, nrows, nrows, nrows, 3 * nrows - 2, iter, 1);
	
	//SequentialCGSolver solver(A, x, b, nrows, nrows, nrows, 3 * nrows - 2, iter, 1);
	// LinearSolver* solver = new SequentialCGSolver(B, x, newB, iter, 1);
	SequentialCGSolver solver(B, x, newB, iter, 1, SEQCG);
	//ParallelCGSolver solver(A, x, b, nrows, nrows, nrows, 3 * nrows - 2, iter, 1);
	//ParallelCGSolver solver(B, x, newB, iter, 1);

	//SequtialBiCGSTABSolver solver(B, x, newB, B.Dimension, B.Dimension, B.Dimension, B.ArrayLength, iter, 1);
	//SequtialBiCGSTABSolver solver(B, x, newB, iter, 1);

	
	//ParallelBiCGSTABSolver solver(B, x, newB, iter, 1);
	//ParallelBiCGSTABSolver solver(A, x, b, nrows, nrows, nrows, 3 * nrows - 2, iter, 1);

	std::cout << "在" << solver.GetClassName() << "中计算求解" << std::endl;
	std::cout << std::endl;
	solver.Process();

	std::cout << std::endl;
	std::cout << "总耗时:" << solver.GetCostTime() << " s" << std::endl;
	std::cout << std::endl;
	//system("pause");
	//delete solver;
 	return 0;
}