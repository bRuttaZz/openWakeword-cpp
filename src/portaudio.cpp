#include <cstdint>
#include <iostream>
#include <queue>
#include <mutex>
#include <vector>
#include <condition_variable>
#include <portaudio.h>
#include "./openwakeword.hpp"


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

static int __audioCallback(const void* inputBuffer, void* outputBuffer, unsigned long framesPerBuffer, const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags, void* userData) {
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

bool oww::openMicStream(oww::AudioQueue& queue, PaStream*& stream, size_t frameSize) {
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "ERROR: error initializing portaudio: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }

    PaStreamParameters inputParams;
    inputParams.device = Pa_GetDefaultInputDevice();
    if (inputParams.device == paNoDevice) {
        std::cerr << "ERROR: no audio-input device found: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }

    inputParams.channelCount = 1;
    inputParams.sampleFormat = paInt16; // s16_le
    inputParams.suggestedLatency = Pa_GetDeviceInfo(inputParams.device)->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;

    err = Pa_OpenStream(
        &stream,
        &inputParams,
        nullptr,
        oww::default_sample_rate,
        frameSize,
        paClipOff,
        __audioCallback,
        &queue
    );
    if (err != paNoError) {
        std::cerr << "ERROR: error opening audio input stream: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }

    return true;
}

bool oww::startMicStream(PaStream* stream) {
    PaError err = Pa_Initialize();
    err = Pa_StartStream(stream);
    if (err != paNoError) {
        std::cerr << "ERROR: failed to start stream: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }
    return true;
}

void oww::stopMicStream(PaStream* stream) {
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();
}
