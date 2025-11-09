#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <format>
#include <iostream>
#include <openwakeword.h>
#include "openwakeword.hpp"

constexpr auto d_log = "Detected: [{}] {}\n";    // Detection-log: %d - model index, %s - model path

void ensureArg(int argc, char *argv[], int argi);
void printUsage(char *argv[]);

int main(int argc, char *argv[]) {
    // default model paths
    char emb_path[] = OWW_RUNTIME_MODEL_PATH_PREFIX "models/embedding_model.onnx";
    char mel_path[] = OWW_RUNTIME_MODEL_PATH_PREFIX "models/melspectrogram.onnx";

    OwwConf conf = {
        NULL, 0, emb_path, mel_path, 0.5f, 4, 20, 4, 0
    };
    char *custom_file = NULL;
    char *usr_cmd = NULL;
    uint verbose = 1;

    // detection log format

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-m" || arg == "--model") {
            ensureArg(argc, argv, i);
            char** new_paths = (char **) realloc(conf.wwd_model_paths, (conf.num_wwd_models + 1) * sizeof(char*));
            if (!new_paths) {
                std::cerr << "Buffer allocation error" << std::endl;
                return -1;
            }
            conf.wwd_model_paths = new_paths;
            conf.wwd_model_paths[conf.num_wwd_models] = strdup(argv[++i]);
            conf.num_wwd_models++;
        } else if (arg == "-f" || arg == "--file") {
            ensureArg(argc, argv, i);
            custom_file = argv[++i];
        } else if (arg == "-e" || arg == "--exec") {
            ensureArg(argc, argv, i);
            usr_cmd = strdup(argv[++i]);
        } else if (arg == "-t" || arg == "--threshold") {
            ensureArg(argc, argv, i);
            conf.threshold = atof(argv[++i]);
        } else if (arg == "-l" || arg == "--trigger-level") {
            ensureArg(argc, argv, i);
            conf.trigger_level = atoi(argv[++i]);
        } else if (arg == "-r" || arg == "--refractory") {
            ensureArg(argc, argv, i);
            conf.refractory = atoi(argv[++i]);
        } else if (arg == "--step-frames") {
            ensureArg(argc, argv, i);
            conf.step_frames = atoi(argv[++i]);
        } else if (arg == "--melspectrogram-model") {
            ensureArg(argc, argv, i);
            conf.melspectrogram_model_path = strdup(argv[++i]);
        } else if (arg == "--embedding-model") {
            ensureArg(argc, argv, i);
            conf.embedding_model_path = strdup(argv[++i]);
        } else if (arg == "--silent") {
            conf.debug = 0;
            verbose = 0;
        } else if (arg == "--debug") {
            conf.debug = 1;
            verbose = 1;
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv);
            exit(0);
        }
    }

    std::FILE *input_file = NULL;
    if (custom_file)
        input_file = std::fopen(custom_file, "rb");

    int status = oww_init(&conf);
    if (status) {
        if (verbose) std::cerr << "Error initiating wakeword context: " << status << std::endl;
        return status;
    }
    if (verbose) std::cerr << "[LOG] Ready" << std::endl;

    status = oww_start_analysis_from_file(input_file);
    if (status) {
        if (verbose) std::cerr << "Error start audio feeding: " << status << std::endl;
        return status;
    }
    while (true) {
        status = oww_wait_for_detection();
        if (status == -1) {
            break;  // stream end
        } else if (status < 0) {
            if (verbose) std::cerr << "Error waiting for detections: " << status << std::endl;
            break;
        }
        if (usr_cmd) {
            if (verbose)
                std::cerr << std::format(d_log, status, conf.wwd_model_paths[status]);
            const int status = system(usr_cmd);
            if (status) return status;
        } else {
            std::cerr << std::format(d_log, status, conf.wwd_model_paths[status]);
        }
    }
    oww_cleanup();
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

    std::cerr << "   -f  FILE  --file           FILE   path to raw pcm data file "
    << std::endl;

    std::cerr << "   -e  CMD   --exec           CMD    Command to be executed on detection time."
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

    std::cerr << "   --step-frames              NUM    number of 80 ms audio chunks to "
            "process at a time (default: 4)"
    << std::endl;

    std::cerr << "   --melspectrogram-model     FILE   path to "
            "melspectrogram.onnx file"
    << std::endl;

    std::cerr << "   --embedding-model          FILE   path to "
            "embedding_model.onnx file"
    << std::endl;

    std::cerr << "   --silent                          disable logging. "
    << std::endl;

    std::cerr << "   --debug                           print model probabilities to stderr"
    << std::endl;

    std::cerr << std::endl;
}

void ensureArg(int argc, char *argv[], int argi) {
    if ((argi + 1) >= argc) {
        printUsage(argv);
        exit(0);
    }
}
