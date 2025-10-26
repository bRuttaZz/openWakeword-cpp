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
 * @return int - 0: on success, negative-integer: on error
 */
int oww_init(const OwwConf *oww_conf);

/**
 * Clean up wwd runtime.
 */
void oww_cleanup();

/**
 * Start wakeword analysis pipeline from a file. Ideal for analysisn stdin
 * @return int - -1: on error, 0: on successfully start the feeding.
 */
int oww_start_analysis_from_file(FILE* file);

/**
 * Start wakeword analysis pipeline from default system microphone.
 * @return int - -1: on error, 0: on successfully start the feeding.
 */
int oww_start_analysis_from_mic();

/**
 * Stop wakeword analysis pipeline
 */
void oww_stop_analysis();

/**
 * Wait for wakeword detection from pipeline.
 * @return int - -1: end of stream. -2: called without starting an analyser, -3: internal error,
 *      0/positive-integer - class name of detected class
 */
int oww_wait_for_detection();



#ifdef __cplusplus
}
#endif

#endif // OPEN_WAKEWORD_CPP
