#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
#include <vector>
#include <queue>

#include <onnxruntime_cxx_api.h>
#include <portaudio.h>

#define OWW_RUNTIME_MODEL_PATH_PREFIX ""

namespace oww { // openwakeword

inline constexpr std::string instanceName = "openWakeWord";
inline constexpr double default_sample_rate = 16000.0;
inline constexpr size_t chunkSamples = 1280; // 80 ms (frame length)
inline constexpr size_t numMels = 32;
inline constexpr size_t embWindowSize = 76; // 775 ms
inline constexpr size_t embStepSize = 8;    // 80 ms
inline constexpr size_t embFeatures = 96;
inline constexpr size_t wwFeatures = 16;

class AudioQueue {
private:
    std::queue<std::vector<std::int16_t>> q;
    size_t max_queue_len;
    std::mutex mtx;
    std::condition_variable cv;

public:
    AudioQueue(size_t queue_len) : max_queue_len(queue_len) {};
    void push(const std::vector<std::int16_t>& data);
    std::vector<std::int16_t> pop();
};

// Configurables
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

// Share process states
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

// Runtime context objects
struct RuntimeContext {
    std::shared_ptr<Settings> settings;
    std::shared_ptr<AudioQueue> micQueue;
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

// Portaudio handler
class PortAudioHandler {
private:
    PaStream* stream;
    PaError err;
    bool started = false;
    bool initialized = false;

public:
    bool openMicStream(oww::AudioQueue &queue, size_t frameSize, bool verbose);
    bool startMicStream();
    void stopMicStream();
};

void feedAudioFromFile(Settings &settings, oww::State &state, std::FILE *inputStream, std::vector<float> &floatSamplesOut);
void feedAudioFromMic(State &state, AudioQueue& queue, std::vector<float> &floatSamplesOut);
void audioToMels(Settings &settings, State &state, std::vector<float> &samplesIn, std::vector<float> &melsOut) ;
void melsToFeatures(Settings &settings, State &state, std::vector<float> &melsIn, std::vector<std::vector<float>> &featuresOut);
void featuresToOutput(Settings &settings, State &state, size_t wwIdx, std::vector<std::vector<float>> &featuresIn, size_t &detections);

} // namespace oww
