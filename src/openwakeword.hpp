#pragma once

#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
#include <vector>

#include <onnxruntime_cxx_api.h>

#define OWW_RUNTIME_MODEL_PATH_PREFIX ""

namespace oww { // openwakeword

inline constexpr std::string instanceName = "openWakeWord";
inline constexpr size_t chunkSamples = 1280; // 80 ms
inline constexpr size_t numMels = 32;
inline constexpr size_t embWindowSize = 76; // 775 ms
inline constexpr size_t embStepSize = 8;    // 80 ms
inline constexpr size_t embFeatures = 96;
inline constexpr size_t wwFeatures = 16;


struct Settings {
    std::filesystem::path melModelPath;
    std::filesystem::path embModelPath;
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
    bool inputStreamExhausted = false;
    bool samplesExhausted = false, melsExhausted = false;
    bool samplesReady = false, melsReady = false;
    std::mutex mutSamples, mutMels, mutReady, mutOutput, mutDetection;
    std::condition_variable cvSamples, cvMels, cvReady, cvDetection, cvStartAnalysis;

    State(size_t numWakeWords);
};

struct RuntimeContext {
    std::shared_ptr<Settings> settings;
    std::shared_ptr<State> state;
    std::vector<std::thread> wwThreads;
    std::thread melThread;
    std::thread featuresThread;
    std::thread inputStreamThread;
    std::vector<float> floatSamples;
    std::vector<float> mels;
    std::vector<std::vector<float>> features;
    size_t detection;
};

void feedAudio(oww::Settings &settings, oww::State &state, std::FILE *inputStream, std::vector<float> &floatSamplesOut);
void audioToMels(Settings &settings, State &state, std::vector<float> &samplesIn, std::vector<float> &melsOut) ;
void melsToFeatures(Settings &settings, State &state, std::vector<float> &melsIn, std::vector<std::vector<float>> &featuresOut);
void featuresToOutput(Settings &settings, State &state, size_t wwIdx, std::vector<std::vector<float>> &featuresIn, size_t &detections);

} // namespace oww
