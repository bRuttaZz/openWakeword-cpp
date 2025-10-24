#include <thread>
#include <iostream>
#include "./openwakeword.hpp"

void ensureArg(int argc, char *argv[], int argi);
void printUsage(char *argv[]);

int main(int argc, char *argv[]) {

  // Re-open stdin/stdout in binary mode
    freopen(NULL, "rb", stdin);

    oww::Settings settings;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-m" || arg == "--model") {
            ensureArg(argc, argv, i);
            settings.wwModelPaths.push_back(std::filesystem::path(argv[++i]));
        } else if (arg == "-t" || arg == "--threshold") {
            ensureArg(argc, argv, i);
            settings.threshold = atof(argv[++i]);
        } else if (arg == "-l" || arg == "--trigger-level") {
            ensureArg(argc, argv, i);
            settings.triggerLevel = atoi(argv[++i]);
        } else if (arg == "-r" || arg == "--refractory") {
            ensureArg(argc, argv, i);
            settings.refractory = atoi(argv[++i]);
        } else if (arg == "--step-frames") {
            ensureArg(argc, argv, i);
            settings.stepFrames = atoi(argv[++i]);
        } else if (arg == "--melspectrogram-model") {
            ensureArg(argc, argv, i);
            settings.melModelPath = std::filesystem::path(argv[++i]);
        } else if (arg == "--embedding-model") {
            ensureArg(argc, argv, i);
            settings.embModelPath = std::filesystem::path(argv[++i]);
        } else if (arg == "--debug") {
            settings.debug = true;
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv);
            exit(0);
        }
    }

    if (settings.wwModelPaths.empty()) {
        std::cerr << "[ERROR] --model is required" << std::endl;
        return 1;
    }

    settings.frameSize = settings.stepFrames * oww::chunkSamples;

    // Absolutely critical for performance
    settings.options.SetIntraOpNumThreads(1);
    settings.options.SetInterOpNumThreads(1);

    const size_t numWakeWords = settings.wwModelPaths.size();
    oww::State state(numWakeWords);

    std::vector<float> floatSamples;
    std::vector<float> mels;
    std::vector<std::vector<float>> features(numWakeWords);

    std::thread melThread(oww::audioToMels, std::ref(settings), std::ref(state), std::ref(floatSamples),
                    std::ref(mels));
    std::thread featuresThread(oww::melsToFeatures, std::ref(settings), std::ref(state), std::ref(mels),
                        std::ref(features));

    std::vector<std::thread> wwThreads;
    for (size_t i = 0; i < numWakeWords; i++) {
        wwThreads.push_back(
            std::thread(oww::featuresToOutput, std::ref(settings),
            std::ref(state), i, std::ref(features))
        );
    }

    // Block until ready
    const size_t numReadyExpected = 2 + numWakeWords;
    {
        std::unique_lock lockReady(state.mutReady);
        state.cvReady.wait(
            lockReady, [&state, numReadyExpected] {
                return state.numReady == numReadyExpected;
            }
        );
    }

    std::cerr << "[LOG] Ready" << std::endl;

    // Main loop
    int16_t samples[settings.frameSize];
    size_t framesRead = std::fread(samples, sizeof(int16_t), settings.frameSize, stdin);

    while (framesRead > 0) {
        {
            std::unique_lock lockSamples{state.mutSamples};

            for (size_t i = 0; i < framesRead; i++) {
                // NOTE: we do NOT normalize here
                floatSamples.push_back((float)samples[i]);
            }

            state.samplesReady = true;
            state.cvSamples.notify_one();
        }
        // Next samples
        framesRead = fread(samples, sizeof(int16_t), settings.frameSize, stdin);
    }

    // Signal mel thread that samples have been exhausted
    {
        std::unique_lock lockSamples{state.mutSamples};
        state.samplesExhausted = true;
        state.samplesReady = true;
        state.cvSamples.notify_one();
    }

    melThread.join();

    // Signal features thread that mels have been exhausted
    {
        std::unique_lock lockMels{state.mutMels};
        state.melsExhausted = true;
        state.melsReady = true;
        state.cvMels.notify_one();
    }
    featuresThread.join();

    // Signal wake word threads that features have been exhausted
    for (size_t i = 0; i < numWakeWords; i++) {
        std::unique_lock lockFeatures{state.mutFeatures[i]};
        state.featuresExhausted[i] = true;
        state.featuresReady[i] = true;
        state.cvFeatures[i].notify_one();
    }

    for (size_t i = 0; i < numWakeWords; i++) {
        wwThreads[i].join();
    }

    return 0;
}

void printUsage(char *argv[]) {
    std::cerr << std::endl;
    std::cerr << "usage: " << argv[0] << " [options]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "options:" << std::endl;
    std::cerr << "   -h        --help                  show this message and exit"
        << std::endl;
    std::cerr << "   -m  FILE  --model          FILE   path to wake word model "
            "(repeat "
            "for multiple models)"
        << std::endl;
    std::cerr << "   -t  NUM   --threshold      NUM    threshold for activation (0-1, "
            "default: 0.5)"
        << std::endl;
    std::cerr << "   -l  NUM   --trigger-level  NUM    number of activations before "
            "output (default: 4)"
        << std::endl;
    std::cerr << "   -r  NUM   --refractory     NUM    number of steps after "
            "activation to wait (default: 20)"
        << std::endl;
    std::cerr
        << "   --step-frames              NUM    number of 80 ms audio chunks to "
            "process at a time (default: 4)"
        << std::endl;
    std::cerr << "   --melspectrogram-model     FILE   path to "
            "melspectrogram.onnx file"
        << std::endl;
    std::cerr << "   --embedding-model          FILE   path to "
            "embedding_model.onnx file"
        << std::endl;
    std::cerr << "   --debug                           print model probabilities to "
            "stderr"
        << std::endl;
    std::cerr << std::endl;
}

void ensureArg(int argc, char *argv[], int argi) {
    if ((argi + 1) >= argc) {
        printUsage(argv);
        exit(0);
    }
}
