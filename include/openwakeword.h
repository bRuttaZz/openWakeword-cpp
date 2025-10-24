#include <stdio.h>
#include <stdint.h>


#ifndef OPEN_WAKEWORD_CPP
#define OPEN_WAKEWORD_CPP

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char** wwd_model_paths;                     // path to wake word models. Ow_init copies the strings internally, caller may free after init
    size_t num_wwd_models;                      // number of wake word models

    char* embedding_model_path;                 // path to embedding_model.onnx file
    char* melspectrogram_model_path;            // path to melspectrogram.onnx file

    float threshold;                            // threshold for activation (0-1, default: 0.5)
    uint8_t trigger_level;                      // number of activations before output (default: 4)
    uint8_t refractory;                         // number of steps after activation to wait (default: 20)
    uint8_t step_frames;                        // number of 80 ms audio chunks to process at a time (default: 4)
    uint8_t debug;                              // print model probabilities to stderr
} OwwConf;


/**
 * Initialize wwd runtime with given configs
 */
int oww_init(const OwwConf *oww_conf);

/**
 * Clean up wwd runtime.
 */
void oww_cleanup();

/**
 * Wait for wakeword invocation from microphone. Uses portaudio to get microphone input.
 * @return int - 1: on successful model invocation, 0: for initialization errors, -1: for microphone device indentifcation errors
 */
int oww_wait_wakeword_from_mic();

/**
 * Wait for wakeword invocation from file description.
 * @param file - file descriptor to read PCM data frames from
 * @return int - 1: on successful model invocation, 0: for initialization errors, -1: file read errors
 */
int oww_wait_wakeword_from_file(FILE* file);


#ifdef __cplusplus
}
#endif

#endif // OPEN_WAKEWORD_CPP
