# OpenWakeWord-CPP Runtime
**A C library / C++ runtime for [openwakeword](https://github.com/dscripka/openWakeWord)**

Rewrite of [rhasspy/openWakeWord-cpp/](https://github.com/rhasspy/openWakeWord-cpp).
Implementation of [dscripka/openWakeWord](https://github.com/dscripka/openWakeWord).

### ToDo
- [ ] Implement portaudio as input stream

### Input Audio Specs (fmt: PCM)
* sample rate: 16000 Hz  (16kHz)
* bit depth: 16-bit 
* channels: 1
* minimal frame length: 80 ms
* bit format: s16_le

### C Library headers
- [openwakeword.h](./include/openwakeword.h)

### CLI
```sh
usage: ./build/openwakeword [options]

options:
   -h        --help                  show this message and exit
   -m  FILE  --model          FILE   path to wake word model (repeat for multiple models)
   -f  FILE  --file           FILE   path to raw pcm data file
   -e  CMD   --exec           CMD    Command to be executed on detection time.
   -t  NUM   --threshold      NUM    threshold for activation (0-1, default: 0.5)
   -l  NUM   --trigger-level  NUM    number of activations before output (default: 4)
   -r  NUM   --refractory     NUM    number of steps after activation to wait (default: 20)
   --step-frames              NUM    number of 80 ms audio chunks to process at a time (default: 4)
   --melspectrogram-model     FILE   path to melspectrogram.onnx file
   --embedding-model          FILE   path to embedding_model.onnx file
   --silent                          disable logging.
   --debug                           print model probabilities to stderr

```

### Sample usage
```sh
# read from ALSA input (for Linux Ofc)
arecord -r 16000 -c 1 -f S16_LE -t raw - | openwakeword --model misc/sheila_v2.onnx
# or using pipwire
pw-cat --record --format s16 --channels 1 --rate 16000 - | ./build/openwakeword --model misc/sheila_v2.onnx
# I mean feel free to use any feeder

# or exececute a command on detection time instead
arecord -r 16000 -c 1 -f S16_LE -t raw - | openwakeword --model misc/sheila_v2.onnx --silent -e "echo 'I aint Jarvis'"
```

Caveats
- For the time being, there is no automation script for building the project and downloading he model assets.
- For pretrained/custom wakeword models, visit [rhasspy/openWakeWord-cpp/](https://github.com/rhasspy/openWakeWord-cpp)

### Build
```sh
# use the make file for orchestration & cmake for compilation

# to get all targets
make

# generate prod build
make package
```
