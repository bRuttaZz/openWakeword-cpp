#include <cstdint>
#include <iostream>
#include <queue>
#include <mutex>
#include <string>
#include <vector>
#include <condition_variable>
#include <portaudio.h>
#include "./openwakeword.hpp"

#ifdef _WIN32
#   define NULL_DEVICE "NUL"
#else
#   define NULL_DEVICE "/dev/null"
#endif


static int _audioCallback(const void* inputBuffer, void* outputBuffer, unsigned long framesPerBuffer, const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags, void* userData) {
    (void)outputBuffer;
    (void)timeInfo;
    (void)statusFlags;

    oww::AudioQueue* audioQueue = static_cast<oww::AudioQueue*>(userData);
    const int16_t* in = static_cast<const int16_t*>(inputBuffer);

    if (inputBuffer == nullptr)
        return paContinue;

    std::vector<int16_t> frame(in, in + framesPerBuffer);
    audioQueue->push(frame);

    return paContinue;
}

void oww::AudioQueue::push(const std::vector<std::int16_t>& data) {
    std::lock_guard<std::mutex> lock(mtx);
    if (q.size() == max_queue_len)
        q.pop();
    q.push(data);
    cv.notify_one();
}

std::vector<std::int16_t> oww::AudioQueue::pop() {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [&]{ return !q.empty(); });
    auto data = q.front();
    q.pop();
    return data;
}

bool oww::PortAudioHandler::init_context(bool verbose=false) {
    if (initialized) {
        return true;
    }
    // aah yeah some dirty log suppression
    auto old_stderr = dup(fileno(stderr));
    if (!verbose) {
        freopen(NULL_DEVICE, "w", stderr);
    }

    err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "ERROR: error initializing portaudio: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }
    initialized = true;

    fflush(stderr);
    if (!verbose) {
        dup2(old_stderr, fileno(stderr));
        close(old_stderr);
    }
    return true;
}

void oww::PortAudioHandler::terminate_context() {
    if (!initialized) return;
    stopMicStream();
    Pa_Terminate();
    initialized = false;
}

bool oww::PortAudioHandler::openMicStream(oww::AudioQueue &queue, size_t frameSize, int16_t device_id=-1, bool verbose=false) {
    if (!init_context(verbose))
        return false;

    PaStreamParameters inputParams;
    if (device_id<0) {
        inputParams.device = Pa_GetDefaultInputDevice();
        if (inputParams.device == paNoDevice) {
            std::cerr << "ERROR: no input device found on machine!" << std::endl;
            return false;
        }
    }
    else
        inputParams.device = device_id;

    const PaDeviceInfo* devInfo = Pa_GetDeviceInfo(inputParams.device);
    if (!devInfo) {
        std::cerr << "ERROR: invalid device ID: " << inputParams.device << std::endl;
        return false;
    }

    inputParams.channelCount = 1;
    inputParams.sampleFormat = paInt16;
    inputParams.suggestedLatency = devInfo->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;

    stream = nullptr;
    err = Pa_OpenStream(
        &stream,
        &inputParams,
        nullptr,
        oww::default_sample_rate,
        frameSize,
        paNoFlag,
        _audioCallback,
        &queue
    );
    if (err != paNoError) {
        std::cerr << "ERROR: error opening audio input stream: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }

    return true;
}

std::vector<std::string> oww::PortAudioHandler::getInputDeviceList() {
    std::vector<std::string> devices;
    init_context(false);

    int count = Pa_GetDeviceCount();
    for (int i=0; i < count; i++) {
        const PaDeviceInfo *info = Pa_GetDeviceInfo(i);
        const PaHostApiInfo *host = Pa_GetHostApiInfo(info->hostApi);

        if (info->maxInputChannels > 0) {
            std::string entry = std::to_string(i) + ": [" + host->name + "] " + info->name +
                " (channels=" + std::to_string(info->maxInputChannels) + ")";
            devices.push_back(entry);
        }
    }
    return devices;
}

bool oww::PortAudioHandler::startMicStream() {
    if (started) {
        std::cerr << "ERROR: portaudio stream already started!" << std::endl;
    }
    err = Pa_StartStream(stream);
    if (err != paNoError) {
        std::cerr << "ERROR: failed to start stream: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }
    started = true;
    return true;
}

void oww::PortAudioHandler::stopMicStream() {
    if (initialized) {
        if (started)
            Pa_StopStream(stream);
        if (stream)
            Pa_CloseStream(stream);
    }
    started = false;
}
