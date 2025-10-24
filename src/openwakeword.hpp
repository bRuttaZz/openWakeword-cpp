#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace oww { // openwakeword

inline constexpr std::string instanceName = "openWakeWord";
inline constexpr size_t chunkSamples = 1280; // 80 ms
inline constexpr size_t numMels = 32;
inline constexpr size_t embWindowSize = 76; // 775 ms
inline constexpr size_t embStepSize = 8;    // 80 ms
inline constexpr size_t embFeatures = 96;
inline constexpr size_t wwFeatures = 16;


struct Settings {
    std::filesystem::path melModelPath = std::filesystem::path("models/melspectrogram.onnx");
    std::filesystem::path embModelPath = std::filesystem::path("models/embedding_model.onnx");
    std::vector<std::filesystem::path> wwModelPaths;

    size_t frameSize = 4 * chunkSamples;
    size_t stepFrames = 4;

    float threshold = 0.5f;
    int triggerLevel = 4;
    int refractory = 20;

    bool debug = false;

    Ort::SessionOptions options;
};

struct State {
    Ort::Env env;
    std::vector<std::mutex> mutFeatures;
    std::vector<std::condition_variable> cvFeatures;
    std::vector<bool> featuresExhausted;
    std::vector<bool> featuresReady;
    size_t numReady;
    bool samplesExhausted = false, melsExhausted = false;
    bool samplesReady = false, melsReady = false;
    std::mutex mutSamples, mutMels, mutReady, mutOutput;
    std::condition_variable cvSamples, cvMels, cvReady;

    State(size_t numWakeWords)
        :   mutFeatures(numWakeWords), cvFeatures(numWakeWords),
            featuresExhausted(numWakeWords), featuresReady(numWakeWords),
            numReady(0), samplesExhausted(false), melsExhausted(false),
            samplesReady(false), melsReady(false)
    {
        env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());
        env.DisableTelemetryEvents();
        std::fill(featuresExhausted.begin(), featuresExhausted.end(), false);
        std::fill(featuresReady.begin(), featuresReady.end(), false);
    }
};

void audioToMels(Settings &settings, State &state, std::vector<float> &samplesIn, std::vector<float> &melsOut) ;
void melsToFeatures(Settings &settings, State &state, std::vector<float> &melsIn, std::vector<std::vector<float>> &featuresOut);
void featuresToOutput(Settings &settings, State &state, size_t wwIdx, std::vector<std::vector<float>> &featuresIn);

} // namespace oww
