#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <openwakeword.h>
#include "./openwakeword.hpp"

static oww::RuntimeContext ctx;

oww::State::State(size_t numWakeWords)
    :mutFeatures(numWakeWords), cvFeatures(numWakeWords),
    featuresExhausted(numWakeWords), featuresReady(numWakeWords),
    numReady(0), samplesExhausted(false), melsExhausted(false),
    samplesReady(false), melsReady(false)
    {
        env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());
        env.DisableTelemetryEvents();
        std::fill(featuresExhausted.begin(), featuresExhausted.end(), false);
        std::fill(featuresReady.begin(), featuresReady.end(), false);
    }


void oww::audioToMels(oww::Settings &settings, oww::State &state, std::vector<float> &samplesIn, std::vector<float> &melsOut) {
    Ort::AllocatorWithDefaultOptions allocator;
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    auto melSession = Ort::Session(state.env, settings.melModelPath.c_str(), settings.options);
    std::vector<int64_t> samplesShape{1, (int64_t)settings.frameSize};

    auto melInputName = melSession.GetInputNameAllocated(0, allocator);
    std::vector<const char *> melInputNames{melInputName.get()};

    auto melOutputName = melSession.GetOutputNameAllocated(0, allocator);
    std::vector<const char *> melOutputNames{melOutputName.get()};

    std::vector<float> todoSamples;

    {
        std::unique_lock lockReady(state.mutReady);
        std::cerr << "[LOG] Loaded mel spectrogram model" << std::endl;
        state.numReady += 1;
        state.cvReady.notify_one();
    }

    while (true) {
        {
            std::unique_lock lockSamples{state.mutSamples};
            state.cvSamples.wait(lockSamples, [&state] { return state.samplesReady; });
            if (state.samplesExhausted && samplesIn.empty())
                break;

            copy(samplesIn.begin(), samplesIn.end(), back_inserter(todoSamples));
            samplesIn.clear();

            if (!state.samplesExhausted)
                state.samplesReady = false;
        }

        while (todoSamples.size() >= settings.frameSize) {
            // Generate mels for audio samples
            std::vector<Ort::Value> melInputTensors;
            melInputTensors.push_back(Ort::Value::CreateTensor<float>(
                memoryInfo, todoSamples.data(), settings.frameSize,
                samplesShape.data(), samplesShape.size())
            );

            auto melOutputTensors = melSession.Run(Ort::RunOptions{nullptr}, melInputNames.data(),
                melInputTensors.data(), melInputNames.size(),
                melOutputNames.data(), melOutputNames.size()
            );

            // (1, 1, frames, mels = 32)
            const auto &melOut = melOutputTensors.front();
            const auto melInfo = melOut.GetTensorTypeAndShapeInfo();
            const auto melShape = melInfo.GetShape();

            const float *melData = melOut.GetTensorData<float>();
            size_t melCount = accumulate(melShape.begin(), melShape.end(), 1, std::multiplies<>());

            {
                std::unique_lock lockMels{state.mutMels};
                for (size_t i = 0; i < melCount; i++) {
                    // Scale mels for Google speech embedding model
                    melsOut.push_back((melData[i] / 10.0f) + 2.0f);
                }
                state.melsReady = true;
                state.cvMels.notify_one();
            }

            todoSamples.erase(todoSamples.begin(), todoSamples.begin() + settings.frameSize);
        }
    }
}

void oww::melsToFeatures(oww::Settings &settings, oww::State &state, std::vector<float> &melsIn, std::vector<std::vector<float>> &featuresOut) {
    Ort::AllocatorWithDefaultOptions allocator;
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    auto embSession = Ort::Session(state.env, settings.embModelPath.c_str(), settings.options);
    std::vector<int64_t> embShape{1, (int64_t)oww::embWindowSize, (int64_t)oww::numMels, 1};

    auto embInputName = embSession.GetInputNameAllocated(0, allocator);
    std::vector<const char *> embInputNames{embInputName.get()};

    auto embOutputName = embSession.GetOutputNameAllocated(0, allocator);
    std::vector<const char *> embOutputNames{embOutputName.get()};

    std::vector<float> todoMels;
    size_t melFrames = 0;

    {
        std::unique_lock lockReady(state.mutReady);
        std::cerr << "[LOG] Loaded speech embedding model" << std::endl;
        state.numReady += 1;
        state.cvReady.notify_one();
    }

    while (true) {
        {
            std::unique_lock lockMels{state.mutMels};
            state.cvMels.wait(lockMels, [&state] { return state.melsReady; });
            if (state.melsExhausted && melsIn.empty())
                break;

            copy(melsIn.begin(), melsIn.end(), back_inserter(todoMels));
            melsIn.clear();

            if (!state.melsExhausted)
                state.melsReady = false;
        }

        melFrames = todoMels.size() / oww::numMels;
        while (melFrames >= oww::embWindowSize) {
            // Generate embeddings for mels
            std::vector<Ort::Value> embInputTensors;
            embInputTensors.push_back(Ort::Value::CreateTensor<float>(
                memoryInfo, todoMels.data(), oww::embWindowSize * oww::numMels, embShape.data(),
                embShape.size())
            );

            auto embOutputTensors = embSession.Run(Ort::RunOptions{nullptr}, embInputNames.data(),
                embInputTensors.data(), embInputTensors.size(),
                embOutputNames.data(), embOutputNames.size()
            );

            const auto &embOut = embOutputTensors.front();
            const auto embOutInfo = embOut.GetTensorTypeAndShapeInfo();
            const auto embOutShape = embOutInfo.GetShape();

            const float *embOutData = embOut.GetTensorData<float>();
            size_t embOutCount = accumulate(embOutShape.begin(), embOutShape.end(), 1, std::multiplies<>());

            // Send to each wake word model
            for (size_t i = 0; i < featuresOut.size(); i++) {
                std::unique_lock lockFeatures{state.mutFeatures[i]};
                copy(embOutData, embOutData + embOutCount, back_inserter(featuresOut[i]));
                state.featuresReady[i] = true;
                state.cvFeatures[i].notify_one();
            }

            // Erase a step's worth of mels
            todoMels.erase(todoMels.begin(), todoMels.begin() + (oww::embStepSize * oww::numMels));
            melFrames = todoMels.size() / oww::numMels;
        }
    }
}

void oww::featuresToOutput(oww::Settings &settings, oww::State &state, size_t wwIdx, std::vector<std::vector<float>> &featuresIn) {
    Ort::AllocatorWithDefaultOptions allocator;
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    auto wwModelPath = settings.wwModelPaths[wwIdx];
    auto wwName = wwModelPath.stem();
    auto wwSession = Ort::Session(state.env, wwModelPath.c_str(), settings.options);

    std::vector<int64_t> wwShape{1, (int64_t)oww::wwFeatures, (int64_t)oww::embFeatures};

    auto wwInputName = wwSession.GetInputNameAllocated(0, allocator);
    std::vector<const char *> wwInputNames{wwInputName.get()};

    auto wwOutputName = wwSession.GetOutputNameAllocated(0, allocator);
    std::vector<const char *> wwOutputNames{wwOutputName.get()};

    std::vector<float> todoFeatures;
    size_t numBufferedFeatures = 0;
    int activation = 0;

    {
        std::unique_lock lockReady(state.mutReady);
        std::cerr << "[LOG] Loaded " << wwName << " model" << std::endl;
        state.numReady += 1;
        state.cvReady.notify_one();
    }

    while (true) {
        {
            std::unique_lock lockFeatures{state.mutFeatures[wwIdx]};
            state.cvFeatures[wwIdx].wait(lockFeatures, [&state, wwIdx] { return state.featuresReady[wwIdx]; });
            if (state.featuresExhausted[wwIdx] && featuresIn[wwIdx].empty())
                break;

            copy(featuresIn[wwIdx].begin(), featuresIn[wwIdx].end(), back_inserter(todoFeatures));
            featuresIn[wwIdx].clear();

            if (!state.featuresExhausted[wwIdx])
                state.featuresReady[wwIdx] = false;
        }

        numBufferedFeatures = todoFeatures.size() / oww::embFeatures;
        while (numBufferedFeatures >= oww::wwFeatures) {
            std::vector<Ort::Value> wwInputTensors;
            wwInputTensors.push_back(Ort::Value::CreateTensor<float>(
                memoryInfo, todoFeatures.data(), oww::wwFeatures * oww::embFeatures,
                wwShape.data(), wwShape.size())
            );

            auto wwOutputTensors = wwSession.Run(
                Ort::RunOptions{nullptr}, wwInputNames.data(),
                wwInputTensors.data(), 1, wwOutputNames.data(), 1
            );

            const auto &wwOut = wwOutputTensors.front();
            const auto wwOutInfo = wwOut.GetTensorTypeAndShapeInfo();
            const auto wwOutShape = wwOutInfo.GetShape();
            const float *wwOutData = wwOut.GetTensorData<float>();
            size_t wwOutCount = accumulate(wwOutShape.begin(), wwOutShape.end(), 1, std::multiplies<>());

            for (size_t i = 0; i < wwOutCount; i++) {
                auto probability = wwOutData[i];
                if (settings.debug) {
                    {
                        std::unique_lock lockOutput(state.mutOutput);
                        std::cerr << wwName << " " << probability << std::endl;
                    }
                }

                if (probability > settings.threshold) {
                    // Activated
                    activation++;
                    if (activation >= settings.triggerLevel) {
                        // Trigger level reached
                        {
                            std::unique_lock lockOutput(state.mutOutput);
                            std::cout << wwName << std::endl;
                        }
                        activation = -settings.refractory;
                    }
                } else {
                    // Back towards 0
                    if (activation > 0) {
                        activation = std::max(0, activation - 1);
                    } else {
                        activation = std::min(0, activation + 1);
                    }
                }
            }

            // Remove 1 embedding
            todoFeatures.erase(todoFeatures.begin(), todoFeatures.begin() + (1 * oww::embFeatures));
            numBufferedFeatures = todoFeatures.size() / oww::embFeatures;
        }
    }
}

extern "C" {
    int oww_init(const OwwConf *oww_conf) {
        if (!oww_conf) return -1;
        if (!oww_conf->melspectrogram_model_path || !oww_conf->embedding_model_path)
            return -1;

        // Re-open stdin/stdout in binary mode
        freopen(NULL, "rb", stdin);

        std::string melModelPathS(oww_conf->melspectrogram_model_path);
        std::string embModelPathS(oww_conf->embedding_model_path);
        std::vector<std::filesystem::path> wwModelPaths;
        for (size_t i=0; i<oww_conf->num_wwd_models; i++) {
            if (!oww_conf->wwd_model_paths[i])
                continue;
            std::cout<< "File: "<< oww_conf->wwd_model_paths[i] << std::endl;
            std::string path(oww_conf->wwd_model_paths[i]);
            wwModelPaths.emplace_back(path);
        }

        ctx.settings = std::make_shared<oww::Settings>();
        ctx.settings->melModelPath = std::filesystem::path(melModelPathS);
        ctx.settings->embModelPath = std::filesystem::path(embModelPathS);
        ctx.settings->wwModelPaths = std::move(wwModelPaths);

        ctx.settings->stepFrames = oww_conf->step_frames;
        ctx.settings->frameSize = ctx.settings->stepFrames * oww::chunkSamples;
        ctx.settings->threshold = oww_conf->threshold;
        ctx.settings->triggerLevel = oww_conf->trigger_level;
        ctx.settings->refractory = oww_conf->refractory;
        ctx.settings->debug = (oww_conf->debug != 0);

        // Absolutely critical for performance
        ctx.settings->options.SetIntraOpNumThreads(1);
        ctx.settings->options.SetInterOpNumThreads(1);

        const size_t numWakeWords = ctx.settings->wwModelPaths.size();
        ctx.state = std::make_shared<oww::State>(numWakeWords);

        // initialize the detection threads
        ctx.floatSamples.clear();
        ctx.mels.clear();
        ctx.features.resize(numWakeWords);

        ctx.melThread = std::thread(oww::audioToMels, std::ref(*ctx.settings), std::ref(*ctx.state), std::ref(ctx.floatSamples), std::ref(ctx.mels));
        ctx.featuresThread = std::thread(oww::melsToFeatures, std::ref(*ctx.settings), std::ref(*ctx.state), std::ref(ctx.mels), std::ref(ctx.features));

        ctx.wwThreads.clear();
        for (size_t i = 0; i < numWakeWords; i++) {
            ctx.wwThreads.emplace_back(
                oww::featuresToOutput, std::ref(*ctx.settings), std::ref(*ctx.state), i, std::ref(ctx.features)
            );
        }

        // Block until ready
        const size_t numReadyExpected = 2 + numWakeWords;
        {
            std::unique_lock lockReady(ctx.state->mutReady);
            std::shared_ptr<oww::State> state = ctx.state;
            state->cvReady.wait(
                lockReady, [&state, numReadyExpected] {
                    return state->numReady == numReadyExpected;
                }
            );
        }
        std::cerr << "[LOG] Ready" << std::endl;

        return 0;
    }

    void oww_cleanup() {
        // Signal mel thread that samples have been exhausted
        {
            std::unique_lock lockSamples{ctx.state->mutSamples};
            ctx.state->samplesExhausted = true;
            ctx.state->samplesReady = true;
            ctx.state->cvSamples.notify_one();
        }
        ctx.melThread.join();

        // Signal features thread that mels have been exhausted
        {
            std::unique_lock lockMels{ctx.state->mutMels};
            ctx.state->melsExhausted = true;
            ctx.state->melsReady = true;
            ctx.state->cvMels.notify_one();
        }
        ctx.featuresThread.join();

        size_t numWakeWords = ctx.settings->wwModelPaths.size();
        // Signal wake word threads that features have been exhausted
        for (size_t i = 0; i < numWakeWords; i++) {
            std::unique_lock lockFeatures{ctx.state->mutFeatures[i]};
            ctx.state->featuresExhausted[i] = true;
            ctx.state->featuresReady[i] = true;
            ctx.state->cvFeatures[i].notify_one();
        }

        for (size_t i = 0; i < numWakeWords; i++) {
            ctx.wwThreads[i].join();
        }

        ctx.state.reset();
        ctx.settings.reset();
    }

    int oww_wait_wakeword_from_file(FILE *file) {
        if (!file)
            file = stdin;

        std::vector<int16_t> samples(ctx.settings->frameSize);
        size_t framesRead = std::fread(samples.data(), sizeof(int16_t), ctx.settings->frameSize, file);
        while (framesRead > 0) {
            {
                std::unique_lock lockSamples{ctx.state->mutSamples};
                for (size_t i = 0; i < framesRead; i++) {
                    // NOTE: we do NOT normalize here
                    ctx.floatSamples.push_back(static_cast<float>(samples[i]));
                }
                ctx.state->samplesReady = true;
                ctx.state->cvSamples.notify_one();
            }
            framesRead = std::fread(samples.data(), sizeof(int16_t), ctx.settings->frameSize, file);
        }

        return 1;
    }
}
