//
// Created by 19528 on 2024/6/28.
//
#include "linmdtw.h"
#include "dynseqalign.h"
#include "dtw.h"

//将二维vector转换为一维array
template <typename T>
T* to_1d_array(vector<vector<T>>& matrix) {
    T* array = new T[matrix.size() * matrix[0].size()];
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            array[i * matrix[0].size() + j] = matrix[i][j];
        }
    }
    return array;
}

template <typename T>
const T* to_1d_array(const vector<vector<T>>& matrix) {
    T* array = new T[matrix.size() * matrix[0].size()];
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            array[i * matrix[0].size() + j] = matrix[i][j];
        }
    }
    return array;
}

//// DTW 函数
DTWBruteResult DTW(const vector<vector<float>>& X, const vector<vector<float>>& Y, int debug) {
    int M = X.size();
    int N = Y.size();
    int d = Y[0].size();

    vector<vector<int>> P(M, vector<int>(N, 0));
    vector<vector<float>> U, L, UL;

    if (debug == 1) {
        U.resize(M, vector<float>(N, 0.0));
        L.resize(M, vector<float>(N, 0.0));
        UL.resize(M, vector<float>(N, 0.0));
    } else {
        U.resize(1, vector<float>(1, 0.0));
        L.resize(1, vector<float>(1, 0.0));
        UL.resize(1, vector<float>(1, 0.0));
    }

    vector<vector<float>> S(M, vector<float>(N, 0.0));

    vector<float> X_flat(M * d);
    vector<float> Y_flat(N * d);

    // Flatten X and Y matrices
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < d; ++j) {
            X_flat[i * d + j] = X[i][j];
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < d; ++j) {
            Y_flat[i * d + j] = Y[i][j];
        }
    }

    vector<int> P_flat(M * N);
    vector<float> U_flat(M * N);
    vector<float> L_flat(M * N);
    vector<float> UL_flat(M * N);
    vector<float> S_flat(M * N);

    float cost = c_dtw(X_flat.data(), Y_flat.data(), P_flat.data(), M, N, d, debug, U_flat.data(), L_flat.data(), UL_flat.data(), S_flat.data());

    // Unflatten matrices
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            P[i][j] = P_flat[i * N + j];
            S[i][j] = S_flat[i * N + j];
            if (debug == 1) {
                U[i][j] = U_flat[i * N + j];
                L[i][j] = L_flat[i * N + j];
                UL[i][j] = UL_flat[i * N + j];
            }
        }
    }

    DTWBruteResult ret;
    ret.cost = cost;
    ret.P = P;
    if (debug == 1) {
        ret.U = U;
        ret.L = L;
        ret.UL = UL;
    }
    ret.S = S;

    return ret;
}


// DTW Diag Step 函数
void DTW_Diag_Step(vector<float>& d0, vector<float>& d1, vector<float>& d2, vector<float>& csm0, vector<float>& csm1, vector<float>& csm2, const vector<vector<float>>& X, const vector<vector<float>>& Y, int diagLen, const vector<int>& box, int reverse, int i, int debug, vector<vector<float>>& U, vector<vector<float>>& L, vector<vector<float>>& UL, vector<vector<float>>& S) {
    int dim = X[0].size();
    c_diag_step(d0.data(), d1.data(), d2.data(), csm0.data(), csm1.data(), csm2.data(), const_cast<float*>(X[0].data()), const_cast<float*>(Y[0].data()), dim, diagLen, const_cast<int*>(box.data()), reverse, i, debug, U[0].data(), L[0].data(), UL[0].data(), S[0].data());
}

