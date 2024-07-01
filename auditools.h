//
// Created by 19528 on 2024/6/29.
//

#ifndef DTWC_AUDITOOLS_H
#define DTWC_AUDITOOLS_H


#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <sndfile.h>
#include <fftw3.h>
#include <opencv2/opencv.hpp>
#include <rubberband/RubberBandStretcher.h>
using namespace std;
std::vector<float> load_audio(const std::string& filename, int& sr);
void save_audio(const std::vector<vector<float>>& x, int sr, const std::string& outprefix);
void gaussianFilter1D(cv::Mat& src, cv::Mat& dst, int sigma, int order);
void maximumFilter1D(const cv::Mat& src, cv::Mat& dst, int size);
cv::Mat  get_DLNC0(const std::vector<float>& x, int sr, int hop_length, int lag=10, bool do_plot=false);
std::vector<std::vector<float>> get_mixed_DLNC0_CENS(const std::vector<float>& x, int sr, int hop_length, float lam=0.1) ;
std::vector<std::vector<float>> get_mfcc_mod(const std::vector<float>& x, int sr, int hop_length, int n_mfcc = 44, int drop = 20, int n_fft = 2048);

std::vector<std::vector<float>> stretch_audio(const std::vector<float>& x1, const std::vector<float>& x2, int sr, std::vector<std::pair<int, int>>& path, int hop_length, bool refine=true);
#endif //DTWC_AUDITOOLS_H
