//
// Created by 19528 on 2024/6/28.
//

#ifndef DTWC_DTW_H
#define DTWC_DTW_H

#include <math.h>
#include <float.h>

#define LEFT 0
#define UP 1
#define DIAG 2
float c_dtw(float* X, float* Y, int* P, int M, int N, int d, int debug, float* U, float* L, float* UL, float* S);
void c_diag_step(float* d0, float* d1, float* d2, float* csm0, float* csm1, float* csm2, float* X, float* Y, int dim, int diagLen, int* box, int reverse, int i, int debug, float* U, float* L, float* UL, float* S);


#endif //DTWC_DTW_H
