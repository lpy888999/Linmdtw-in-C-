//
// Created by 19528 on 2024/6/28.
//

#ifndef DTWC_DYNSEQALIGN_H
#define DTWC_DYNSEQALIGN_H


DTWBruteResult DTW(const vector<vector<float>>& X, const vector<vector<float>>& Y, int debug);
void DTW_Diag_Step(vector<float>& d0, vector<float>& d1, vector<float>& d2, vector<float>& csm0, vector<float>& csm1, vector<float>& csm2, const vector<vector<float>>& X, const vector<vector<float>>& Y, int diagLen, const vector<int>& box, int reverse, int i, int debug, vector<vector<float>>& U, vector<vector<float>>& L, vector<vector<float>>& UL, vector<vector<float>>& S);

#endif //DTWC_DYNSEQALIGN_H
