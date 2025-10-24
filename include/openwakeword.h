#include <stdio.h>

#ifndef OPEN_WAKEWORD_CPP
#define OPEN_WAKEWORD_CPP

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Wait for wakeword invocation from microphone. Uses portaudio to get microphone input.
 * @param threshold - wakeword detection threshold, between 0 and 1
 * @param wd_model_path - wakeword detection onnx model path.
 * @param embedding_model_path - embedding model path. Uses the default model path if provided as none.
 * @param melspectrogram_model_path - melspectrogram model path. Uses the default model path if provided as none.
 * @return int - 1: on successful model invocation, 0: for initialization errors, -1: for microphone device indentifcation errors
 */
int wait_wakeword_from_mic(float threshold, const char* wd_model_path, const char* embedding_model_path, const char* melspectrogram_model_path);

/**
 * Wait for wakeword invocation from file description.
 * @param file - file descriptor to read PCM data frames from
 * @param threshold - wakeword detection threshold, between 0 and 1
 * @param wd_model_path - wakeword detection onnx model path.
 * @param embedding_model_path - embedding model path. Uses the default model path if provided as none.
 * @param melspectrogram_model_path - melspectrogram model path. Uses the default model path if provided as none.
 * @return int - 1: on successful model invocation, 0: for initialization errors, -1: file read errors
 */
int wait_wakeword_from_file(FILE* file, float threshold, const char* wd_model_path, const char* embedding_model_path, const char* melspectrogram_model_path);


#ifdef __cplusplus
}
#endif

#endif // OPEN_WAKEWORD_CPP
