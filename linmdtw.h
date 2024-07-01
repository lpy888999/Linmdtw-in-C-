#ifndef LINMDTW_H
#define LINMDTW_H
#include <iostream>
#include <algorithm>
#include <limits>
#include <cmath>
#include <tuple>
#include <map>
#include <vector>
#include <cuda_runtime.h>
using namespace std;
// 结果结构体
struct DTWResult {
    float cost;
    std::vector<std::pair<int, int>> path;
};

// 定义结果结构体
struct DTWBruteResult {
    float cost;
    vector<vector<int>> P; // Backpointer matrix
    vector<vector<float>> U, L, UL; // Choice matrices
    vector<vector<float>> S; // Accumulated cost matrix
};

struct DTWDiagResult {
    float cost;
    vector<float> d0, d1, d2;
    vector<float> csm0, csm1, csm2;
    vector<vector<float>> U, L, UL, S, CSM;
};

void check_euclidean_inputs(const std::vector<std::vector<float>>& X, const std::vector<std::vector<float>>& Y);

// 检查输入数据的维度和数据类型

DTWDiagResult dtw_diag(const vector<vector<float>>& X, const vector<vector<float>>& Y, int k_save = -1, int k_stop = -1, vector<int> box = {}, bool reverse = false, bool debug = false, std::map<std::string, long double>* metadata = nullptr);
// brute force DTW 和 对齐路径追溯函数
DTWBruteResult dtw_brute(const vector<vector<float>>& X, const vector<vector<float>>& Y, bool debug = false);
DTWResult dtw_brute_backtrace(const vector<vector<float>>& X, const vector<vector<float>>& Y, bool debug = false);
// linmdtw函数声明
DTWResult linmdtw(vector<vector<float>>& X, vector<vector<float>>& Y, vector<int> box = {}, int min_dim = 100, bool do_gpu = true, std::map<std::string, long double>* metadata = nullptr) ;


#endif // LINMDTW_H
