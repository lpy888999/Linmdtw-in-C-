//
// Created by 19528 on 2024/6/28.
//
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <map>
#include <vector>
#include "linmdtw.h"
#include"auditools.h"

void align_pieces(const std::string& filename1, const std::string& filename2, int sr, int hop_length, bool do_mfcc, bool compare_cpu, bool do_stretch = false) {
    if (!std::ifstream(filename1).good()) {
        std::cerr << "Skipping " << filename1 << std::endl;
        return;
    }
    if (!std::ifstream(filename2).good()) {
        std::cerr << "Skipping " << filename2 << std::endl;
        return;
    }

    std::string prefix = do_mfcc ? "mfcc" : "chroma";
    std::string pathfilename = filename1 + "_" + prefix + "_path.mat";

    if (std::ifstream(pathfilename).good()) {
        std::cout << "Already computed all alignments on " << filename1 << " " << filename2 << std::endl;
        return;
    }

    auto x1 = load_audio(filename1, sr);
    auto x2 = load_audio(filename2, sr);

    std::vector<std::vector<float>> X1, X2;
    if (do_mfcc) {
        X1 = get_mfcc_mod(x1, sr, hop_length);
        X2 = get_mfcc_mod(x2, sr, hop_length);
    } else {
        X1 = get_mixed_DLNC0_CENS(x1, sr, hop_length);
        X2 = get_mixed_DLNC0_CENS(x2, sr, hop_length);
    }

    if (std::ifstream(pathfilename).good()) {
        std::cout << "Already computed full " << prefix << " alignments on " << filename1 << " " << filename2 << std::endl;
    } else {
        auto start_time = std::chrono::high_resolution_clock::now();


        std::map<std::string, long double> metadata = {{"totalCells", 0}, {"M", static_cast<double>(X1.size())}, {"N", static_cast<double>(X2.size())}, {"timeStart", start_time.time_since_epoch().count()}};
        std::cout << "Starting GPU Alignment..." << std::endl;

        DTWResult result = linmdtw(X1, X2, {}, 500, true, &metadata);
        auto end_time = std::chrono::high_resolution_clock::now();
//        std::chrono::duration<double> gpu_duration = end_time - start_time;
//        metadata["time_gpu"] = gpu_duration.count();

        std::ofstream meta_file(filename1 + "_" + prefix + "_stats.json");
        meta_file << "{\n";
        for (auto& kv : metadata) {
            meta_file << "  \"" << kv.first << "\": " << kv.second << ",\n";
        }
        meta_file << "}\n";
        meta_file.close();

        cv::Mat path_gpu_mat(result.path.size(), 2, CV_32S);
        for (size_t i = 0; i < result.path.size(); ++i) {
            path_gpu_mat.at<int>(i, 0) = result.path[i].first;
            path_gpu_mat.at<int>(i, 1) = result.path[i].second;
        }

        cv::FileStorage fs(pathfilename, cv::FileStorage::WRITE);
        fs << "path_gpu" << path_gpu_mat;
        fs.release();

        if (do_stretch) {
            std::cout << "Stretching..." << std::endl;
            auto stretched_audio = stretch_audio(x1, x2, sr, result.path, hop_length);
            save_audio(stretched_audio, sr, filename1 + "_" + prefix + "_sync");
            std::cout << "Finished stretching" << std::endl;
        }

        if (compare_cpu) {
            start_time = std::chrono::high_resolution_clock::now();
            std::cout << "Doing CPU alignment..." << std::endl;
            auto path_cpu = dtw_brute_backtrace(X1, X2);
            end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> cpu_duration = end_time - start_time;
            metadata["time_cpu"] = cpu_duration.count();

            cv::Mat path_cpu_mat(path_cpu.path.size(), 2, CV_32S);
            for (size_t i = 0; i < path_cpu.path.size(); ++i) {
                path_cpu_mat.at<int>(i, 0) = path_cpu.path[i].first;
                path_cpu_mat.at<int>(i, 1) = path_cpu.path[i].second;
            }

            fs.open(pathfilename, cv::FileStorage::APPEND);
            fs << "path_cpu" << path_cpu_mat;
            fs.release();
        }
    }
}
void test1(){
    std::vector<std::vector<float>> X = {
            {1.0, 2.0},
            {2.0, 3.0},
            {2,2},
            {3.0, 4.0},
            {4.0, 5.0},
            {5.0, 6.0},
    };

    std::vector<std::vector<float>> Y = {
            {1.0, 2.0},
            {2.1, 3.1},
            {3.1, 4.1},
            {4.1, 5.1},
            {5.1, 6.1},
            {6,6}
    };

    // 计算DTW
    DTWResult result = linmdtw(X, Y,{},5,true);

    // 输出结果
    std::cout << "Cost: " << result.cost << std::endl;
    std::cout << "Path: ";
    for (const auto& p : result.path) {
        std::cout << "(" << p.first << ", " << p.second << ") ";
    }
    std::cout << std::endl;
}
void test2(){
    std::vector<std::vector<float>> X = {
            {1.0, 2.0},
            {2.0, 3.0},
            {3.0, 4.0},
            {4.0, 5.0},
            {5.0, 6.0}
    };

    std::vector<std::vector<float>> Y = {
            {1.0, 2.0},
            {2.0, 3.0},
            {3.0, 6.0},
            {4.0, 8.0},
            {5.0, 10.0}
    };

    // 计算DTW
    std::map<std::string, long double> metadata;
    DTWResult result = linmdtw(X, Y, {}, 5, true, &metadata);

    // 输出结果
    std::cout << "Cost: " << result.cost << std::endl;
    std::cout << "Path: ";
    for (const auto& p : result.path) {
        std::cout << "(" << p.first << ", " << p.second << ") ";
    }
    std::cout << std::endl;

}
int main() {
    std::string filename1 = "../OrchestralPieces/Short/1_0.mp3";
    std::string filename2 = "../OrchestralPieces/Short/1_1.mp3";
    int hop_length = 512;
    int sr = 22050;
    bool compare_cpu = false;
    bool do_stretch = false;
    align_pieces(filename1, filename2, sr, hop_length, true, compare_cpu, do_stretch);
//     示例数据：两个简单的二维时间序列
//    test1();
//    test2();

    return 0;
}
