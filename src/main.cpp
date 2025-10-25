#include <cstddef>
#include <cstring>
#include <iostream>
#include <openwakeword.h>

void ensureArg(int argc, char *argv[], int argi);
void printUsage(char *argv[]);

int main(int argc, char *argv[]) {

    // default model paths
    char emb_path[] = "models/embedding_model.onnx";
    char mel_path[] = "models/melspectrogram.onnx";

    OwwConf conf = {
        NULL, 0, emb_path, mel_path, 0.5f, 4, 20, 4, 0
    };
    char *custom_file = NULL;

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
        } else if (arg == "--debug") {
            conf.debug = 1;
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
        std::cerr << "Error initiating wakeword context: " << status << std::endl;
        return 1;
    }
    oww_wait_wakeword_from_file(input_file);
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
