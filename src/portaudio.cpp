#include <cstdint>
#include <iostream>
#include <queue>
#include <mutex>
#include <vector>
#include <fstream>
#include <condition_variable>
#include <portaudio.h>
#include "./openwakeword.hpp"

std::ofstream outfile("mic_output.raw", std::ios::binary);

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

    outfile.write(reinterpret_cast<const char*>(frame.data()), frame.size() * sizeof(int16_t));

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


bool oww::PortAudioHandler::openMicStream(oww::AudioQueue &queue, size_t frameSize, bool verbose=false) {
    if (initialized) {
        std::cerr << "ERROR: portaudio context already initialized!" << std::endl;
        return false;
    }
    err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "ERROR: error initializing portaudio: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }

    stream = nullptr;
    err = Pa_OpenDefaultStream(&stream, 1, 0, paInt16, oww::default_sample_rate, frameSize, _audioCallback, &queue);
    if (err != paNoError) {
        std::cerr << "ERROR: error opening audio input stream: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }

    initialized = true;
    return true;
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
        Pa_CloseStream(stream);
        Pa_Terminate();
    }
    started = false;
    initialized = false;
}
