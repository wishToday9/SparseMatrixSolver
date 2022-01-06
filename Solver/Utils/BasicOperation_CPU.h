#pragma once

#include <math.h>
#include "../Main/TypeDef.h"

//æÿ’Û°¡œÚ¡ø
void Matrix_multi_Vector(uint32 nrows, double* ax, double* sra, uint32* clm, uint32* fnz, double* x) {
	for (uint32 i = 0; i < nrows; ++i)
		for (uint32 j = fnz[i]; j < fnz[i + 1]; ++j)
			ax[i] += (sra[j] * x[clm[j]]);
}

//µ„≥À
double dotproduct(uint32 nrows, double* z, double* r) {
	double temp = 0.0;
	for (uint32 i = 0; i < nrows; ++i)
		temp += (z[i] * r[i]);
	return temp;
}

double maxnorm(uint32 nrows, double* r) {
	double temp = abs(r[0]);
	for (uint32 i = 1; i < nrows; ++i)
		if (temp < abs(r[i]))
			temp = abs(r[i]);
	return temp;
}

double squarenorm(uint32 nrows, double* r) {
	double sum = 0;
	for (uint32 i = 0; i < nrows; ++i) {
		sum += pow(r[i], 2);
	}
	return sqrt(sum);
}