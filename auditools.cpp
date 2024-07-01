//
// Created by 19528 on 2024/6/29.
//

#include "auditools.h"
#include "alignmenttools.h"
// Function to load audio file
std::vector<float> load_audio(const std::string& filename, int& sr) {
    std::string wavfilename = filename + ".wav";

    // 使用 ffmpeg 将 mp3 文件转换为 wav 文件
    std::string command = "ffmpeg -i " + filename + " -ar " + std::to_string(sr) + " -ac 1 " + wavfilename;
    int ret = std::system(command.c_str());
    if (ret != 0) {
        std::cerr << "Error: ffmpeg command failed for " << filename << std::endl;
        return {};
    }

    SF_INFO sfinfo;
    SNDFILE* sndfile = sf_open(wavfilename.c_str(), SFM_READ, &sfinfo);
    if (!sndfile) {
        std::cerr << "Error: unable to open audio file " << wavfilename << std::endl;
        return {};
    }

    std::vector<float> audio_data(sfinfo.frames);
    sf_read_float(sndfile, audio_data.data(), sfinfo.frames);
    sr = sfinfo.samplerate;
    sf_close(sndfile);

    // 删除临时的 wav 文件
    std::remove(wavfilename.c_str());

    return audio_data;
}

// Function to save audio file
void save_audio(const std::vector<std::vector<float>>& x, int sr, const std::string& outprefix) {
    std::string wavfilename = outprefix + ".wav";
    SF_INFO sfinfo;
    sfinfo.channels = 1; // Assuming mono audio. Change this if you have stereo.
    sfinfo.samplerate = sr;
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    SNDFILE* sndfile = sf_open(wavfilename.c_str(), SFM_WRITE, &sfinfo);
    if (!sndfile) {
        std::cerr << "Error: unable to open audio file for writing " << wavfilename << std::endl;
        return;
    }

    // Flatten the 2D vector to 1D
    std::vector<float> flattened_audio;
    for (const auto& channel : x) {
        flattened_audio.insert(flattened_audio.end(), channel.begin(), channel.end());
    }

    sf_write_float(sndfile, flattened_audio.data(), flattened_audio.size());
    sf_close(sndfile);
}

// Apply Gaussian filter
cv::Mat gaussian_filter(const cv::Mat& input, double sigma) {
    cv::Mat output;
    cv::GaussianBlur(input, output, cv::Size(0, 0), sigma);
    return output;
}

std::vector<std::vector<float>> matToVector(const cv::Mat& mat) {
    std::vector<std::vector<float>> vec(mat.rows, std::vector<float>(mat.cols));
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            vec[i][j] = mat.at<float>(i, j);
        }
    }
    return vec;
}

// Compute Constant-Q Transform (CQT) - Placeholder
cv::Mat compute_CQT(const std::vector<float>& x, int sr, int hop_length) {
    // Implement or find a library for CQT computation
    // Placeholder: return a dummy matrix
    int rows = 84; // Example number of frequency bins
    int cols = x.size() / hop_length; // Example number of frames
    return cv::Mat::zeros(rows, cols, CV_32F);
}

// Compute CENS features - Placeholder
cv::Mat compute_CENS(const std::vector<float>& x, int sr, int hop_length) {
    // Implement or find a library for CENS computation
    // Placeholder: return a dummy matrix
    int rows = 12; // Example number of chroma bins
    int cols = x.size() / hop_length; // Example number of frames
    return cv::Mat::zeros(rows, cols, CV_32F);
}

// Apply maximum filter
cv::Mat maximum_filter(const cv::Mat& input, int size) {
    cv::Mat output;
    cv::dilate(input, output, cv::Mat::ones(1, size, CV_32F));
    return output;
}

cv::Mat get_DLNC0(const std::vector<float>& x, int sr, int hop_length, int lag, bool do_plot) {
    // Compute CQT
    cv::Mat X = compute_CQT(x, sr, hop_length);

    // Apply Gaussian filter and half-wave rectification
    X = gaussian_filter(X, 5);
    cv::threshold(X, X, 0, 1, cv::THRESH_TOZERO);

    // Retain peaks
    cv::Mat XLeft = X(cv::Rect(0, 0, X.cols - 2, X.rows));
    cv::Mat XRight = X(cv::Rect(2, 0, X.cols - 2, X.rows));
    cv::Mat mask = cv::Mat::zeros(X.size(), X.type());
    mask(cv::Rect(1, 0, X.cols - 2, X.rows)) = (X(cv::Rect(1, 0, X.cols - 2, X.rows)) > XLeft) & (X(cv::Rect(1, 0, X.cols - 2, X.rows)) > XRight);
    X.setTo(0, mask == 0);

    // Fold into octave
    int n_octaves = X.rows / 12;
    cv::Mat X2 = cv::Mat::zeros(12, X.cols, X.type());
    for (int i = 0; i < n_octaves; ++i) {
        X2 += X(cv::Rect(0, i * 12, X.cols, 12));
    }
    X = X2;

    // Compute norms and normalize
    cv::Mat norms;
    cv::reduce(X.mul(X), norms, 0, cv::REDUCE_SUM, CV_32F);
    cv::sqrt(norms, norms);
    norms = maximum_filter(norms, 2 * sr / hop_length);
    // Prevent division by zero by replacing zeros in norms with ones
    norms.setTo(1, norms == 0);

    // Ensure norms has the correct dimensions
    cv::Mat norms_mat;
    cv::repeat(norms, X.rows, 1, norms_mat);
    cv::divide(X, norms, X);

    // Apply DLNC0
    cv::Mat decays = cv::Mat::zeros(1, lag, CV_32F);
    for (int i = 0; i < lag; ++i) {
        decays.at<float>(i) = std::sqrt((lag - i) / static_cast<float>(lag));
    }
    cv::Mat XRet = cv::Mat::zeros(X.rows, X.cols - lag + 1, X.type());
    for (int i = 0; i < lag; ++i) {
        XRet += X(cv::Rect(i, 0, X.cols - lag + 1, X.rows)).mul(decays.at<float>(i));
    }

    return XRet;
}

std::vector<std::vector<float>> get_mixed_DLNC0_CENS(const std::vector<float>& x, int sr, int hop_length, float lam) {
    cv::Mat DLNC0 = get_DLNC0(x, sr, hop_length, 10, false);
    cv::Mat CENS = lam * compute_CENS(x, sr, hop_length);

    cv::Mat result;
    cv::hconcat(DLNC0, CENS, result);

    return matToVector(result);
}

// Function to compute the MFCC-mod features
std::vector<std::vector<float>> compute_mel_filterbank(int sr, int n_fft, int n_mels);
std::vector<float> dct(const std::vector<float>& input, int n_mfcc);

// 计算MFCC-mod特征的函数
std::vector<std::vector<float>> get_mfcc_mod(const std::vector<float>& x, int sr, int hop_length, int n_mfcc, int drop, int n_fft) {
    // 计算帧数
    int n_frames = (x.size() - n_fft) / hop_length + 1;
    if (n_frames <= 0) {
        std::cerr << "Error: Not enough audio data for the given FFT size and hop length." << std::endl;
        return {};
    }

    cv::Mat mfcc_mod(n_frames, n_mfcc - drop, CV_32F);

    // 窗函数 (Hamming window)
    std::vector<float> window(n_fft);
    for (int i = 0; i < n_fft; ++i) {
        window[i] = 0.54 - 0.46 * cos(2 * M_PI * i / (n_fft - 1));
    }

    // FFTW 设置
    fftwf_plan plan;
    fftwf_complex* out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (n_fft / 2 + 1));
    std::vector<float> in(n_fft);
    plan = fftwf_plan_dft_r2c_1d(n_fft, in.data(), out, FFTW_ESTIMATE);

    // 计算梅尔滤波器组
    std::vector<std::vector<float>> mel_filterbank = compute_mel_filterbank(sr, n_fft, 24);

    // 处理每个帧
    for (int i = 0; i < n_frames; ++i) {
        // 应用窗口并执行 FFT
        for (int j = 0; j < n_fft; ++j) {
            in[j] = x[i * hop_length + j] * window[j];
        }
        fftwf_execute(plan);

        // 计算功率谱
        std::vector<float> power_spectrum(n_fft / 2 + 1);
        for (int j = 0; j < n_fft / 2 + 1; ++j) {
            power_spectrum[j] = (out[j][0] * out[j][0] + out[j][1] * out[j][1]) / n_fft;
        }

        // 应用梅尔滤波器组
        std::vector<float> mel_spectrum(24, 0.0f);
        for (int j = 0; j < 24; ++j) {
            for (int k = 0; k < n_fft / 2 + 1; ++k) {
                mel_spectrum[j] += mel_filterbank[j][k] * power_spectrum[k];
            }
            mel_spectrum[j] = std::log(mel_spectrum[j] + 1e-10);  // 计算对数功率谱
        }

        // 计算 MFCC
        std::vector<float> mfcc = dct(mel_spectrum, n_mfcc);

        // 丢弃系数并存储结果
        for (int j = drop; j < n_mfcc; ++j) {
            mfcc_mod.at<float>(i, j - drop) = mfcc[j];
        }
    }

    fftwf_destroy_plan(plan);
    fftwf_free(out);

    std::vector<std::vector<float>> result(n_frames, std::vector<float>(n_mfcc - drop));
    for (int i = 0; i < n_frames; ++i) {
        for (int j = 0; j < n_mfcc - drop; ++j) {
            result[i][j] = mfcc_mod.at<float>(i, j);
        }
    }

    return result;
}
// 用于生成线性间隔向量的辅助函数
std::vector<float> linspace(float start, float end, int num) {
    std::vector<float> result(num);
    float step = (end - start) / (num - 1);
    for (int i = 0; i < num; ++i) {
        result[i] = start + step * i;
    }
    return result;
}
// 计算梅尔滤波器组的函数
std::vector<std::vector<float>> compute_mel_filterbank(int sr, int n_fft, int n_mels) {
    // 梅尔滤波器组计算（这里使用简单的线性间隔梅尔滤波器组）
    std::vector<std::vector<float>> mel_filterbank(n_mels, std::vector<float>(n_fft / 2 + 1, 0.0f));
    // 梅尔频率范围
    float mel_min = 0.0f;
    float mel_max = 2595.0f * std::log10(1.0f + sr / 2.0f / 700.0f);
    std::vector<float> mel_points = linspace(mel_min, mel_max, n_mels + 2);

    std::vector<int> bin_points(n_mels + 2);
    for (int i = 0; i < mel_points.size(); ++i) {
        bin_points[i] = static_cast<int>(n_fft * 700.0f * (std::pow(10.0f, mel_points[i] / 2595.0f) - 1.0f) / sr);
    }

    for (int i = 1; i < n_mels + 1; ++i) {
        for (int j = bin_points[i - 1]; j < bin_points[i]; ++j) {
            mel_filterbank[i - 1][j] = (j - bin_points[i - 1]) / (float)(bin_points[i] - bin_points[i - 1]);
        }
        for (int j = bin_points[i]; j < bin_points[i + 1]; ++j) {
            mel_filterbank[i - 1][j] = (bin_points[i + 1] - j) / (float)(bin_points[i + 1] - bin_points[i]);
        }
    }
    return mel_filterbank;
}

// 离散余弦变换 (DCT-II) 的函数
std::vector<float> dct(const std::vector<float>& input, int n_mfcc) {
    int n = input.size();
    std::vector<float> result(n_mfcc, 0.0f);
    for (int k = 0; k < n_mfcc; ++k) {
        for (int i = 0; i < n; ++i) {
            result[k] += input[i] * std::cos(M_PI * k * (2 * i + 1) / (2 * n));
        }
        if (k == 0) {
            result[k] *= std::sqrt(1.0 / n);
        } else {
            result[k] *= std::sqrt(2.0 / n);
        }
    }
    return result;
}



std::vector<std::vector<float>> stretch_audio(const std::vector<float>& x1, const std::vector<float>& x2, int sr, std::vector<std::pair<int, int>>& path, int hop_length, bool refine) {
    std::cout << "Stretching..." << std::endl;

    std::vector<std::pair<int, int>> path_final = path;
    if (refine) {
        path_final = refine_warping_path(path_final);
    }

    for (auto& p : path_final) {
        p.first *= hop_length;
        p.second *= hop_length;
    }
    path_final.push_back({static_cast<int>(x1.size()), static_cast<int>(x2.size())});

    // Initialize RubberBandStretcher
    RubberBand::RubberBandStretcher stretcher(sr, 1, RubberBand::RubberBandStretcher::OptionProcessRealTime | RubberBand::RubberBandStretcher::OptionStretchPrecise);

    // Prepare input for RubberBand
    std::vector<float> x1_stretch(x2.size());
    const float* input[1] = {x1.data()};
    float* output[1] = {x1_stretch.data()};

    size_t current_index = 0;
    for (const auto& p : path_final) {
        size_t segment_length = p.second - current_index;
        if (segment_length > 0 && p.first + segment_length <= x1.size()) {
            std::vector<float> segment(x1.begin() + p.first, x1.begin() + p.first + segment_length);
            input[0] = segment.data();
            stretcher.process(input, static_cast<int>(segment.size()), false);
            output[0] = x1_stretch.data() + current_index;
            stretcher.retrieve(output, static_cast<int>(segment_length));
            current_index += segment_length;
        }
    }

    // Create output stereo audio
    std::vector<std::vector<float>> x3(std::min(x1_stretch.size(), x2.size()), std::vector<float>(2));
    for (size_t i = 0; i < x3.size(); ++i) {
        x3[i][0] = x1_stretch[i];
        x3[i][1] = x2[i];
    }

    return x3;
}
