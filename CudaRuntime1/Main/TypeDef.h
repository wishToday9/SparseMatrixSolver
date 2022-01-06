#pragma once
#include <string>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
typedef unsigned char             uint8;
typedef unsigned short            uint16;
typedef unsigned int              uint32;
typedef unsigned long long int    uint64;
typedef signed char               int8;
typedef short                     int16;
typedef int                       int32;
typedef signed long long int      int64;
enum ClassType
{
	SEQBASE,
	PARBASE,
	SEQCG,
	PARCG,
	SEQBICG,
	PARBICG
};

enum StoreType
{
	COOTYPE,
	CSRTYPE
};

const std::string className[] = { "LinearSolver", "ParallelSolver", "SequentialCGSolver", "ParallelCGSolver", "SequtialBiCGSTABSolver", "ParallelBiCGSTABSolver"};

//COO存储格式
struct COO
{
	uint32* RowIndex;
	uint32* ColIndex;
	double* Adata;
	~COO(){
		//delete[] RowIndex;
		//RowIndex = nullptr;

		//delete[] ColIndex;
		//ColIndex = nullptr;

		//delete[] Adata;
		//Adata = nullptr;
	}
};


//CSR存储格式
struct CSR
{
	uint32* RowOffset;
	uint32* ColIndex;
	double* Adata;
	uint32 Dimension;
	uint32 ArrayLength;
	CSR(const CSR& temp) noexcept {
		//std::cout << "copy" << std::endl;
		RowOffset = new uint32[temp.Dimension + 1];
		ColIndex = new uint32[temp.ArrayLength];
		Adata = new double[temp.ArrayLength];
		memcpy(RowOffset, temp.RowOffset, (temp.Dimension + 1) * sizeof(uint32));
		memcpy(ColIndex, temp.ColIndex, temp.ArrayLength * sizeof(uint32));
		memcpy(Adata, temp.Adata, temp.ArrayLength * sizeof(double));

		Dimension = temp.Dimension;
		ArrayLength = temp.ArrayLength;
	}
	CSR(CSR&& temp) noexcept{
		//std::cout << "move" << std::endl;
		RowOffset = temp.RowOffset;
		ColIndex = temp.ColIndex;
		Adata = temp.Adata;
		Dimension = temp.Dimension;
		ArrayLength = temp.ArrayLength;

		temp.RowOffset = nullptr;
		temp.ColIndex = nullptr;
		temp.Adata = nullptr;
		temp.Dimension = 0;
		temp.ArrayLength = 0;
	}

	~CSR() {
		//std::cout << "析构CSR" << std::endl;
		delete[] RowOffset;
		RowOffset = nullptr;

		delete[] ColIndex;
		ColIndex = nullptr;

		delete[] Adata;
		Adata = nullptr;
	}
	CSR() {
		
	}
	CSR(std::string path) {
		//pername ton pinaka
		std::ifstream file(path);
		if (!file.is_open()) {
			std::cout << "open failed!" << std::endl;
		}
		std::string str;
		int k = 0, nnz = 0, size = 0;
		int n; // proti grammh - size
		getline(file, str);
		std::stringstream stream(str);
		stream >> n;
		nnz = n;
		//cout<<nnz<<endl;
		getline(file, str);   // deuterh grammh gia to size  
		std::stringstream stream2(str);
		stream2 >> n;
		size = n;
		//cout<<size<<endl;
		Dimension = size;
		ArrayLength = nnz;
		RowOffset = new uint32[size + 1];
		ColIndex = new uint32[nnz];
		Adata = new double[nnz];

		getline(file, str);   // trith grammh gia to val[]  
		std::stringstream stream3(str);
		double v;
		int i = 0;
		//std::cout << "val:";
		while (stream3 >> v) {
			if (!stream) break;
			Adata[i] = v;
			//std::cout << Adata[i] << "    ";
			i++;
		}
		std::cout << std::endl;
		getline(file, str);   // 4h grammh gia to indj[]  
		std::stringstream stream4(str);
		i = 0;
		//std::cout << "indj:";
		while (stream4 >> n) {
			if (!stream) break;
			ColIndex[i] = n;
			//std::cout << ColIndex[i] << "   ";
			i++;
		}
		std::cout << std::endl;
		getline(file, str);   // 5h grammh gia to indi[]  
		std::stringstream stream5(str);
		i = 0;
		while (stream5 >> n) {
			if (!stream) break;
			RowOffset[i] = n;
			//std::cout << RowOffset[i] << "    ";
			i++;
		}
	}
};
