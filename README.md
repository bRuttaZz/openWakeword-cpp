# OpenWakeWord-CPP Runtime
**A C library / C++ runtime for [openwakeword](https://github.com/dscripka/openWakeWord)**

Elaborated rewrite of [rhasspy/openWakeWord-cpp/](https://github.com/rhasspy/openWakeWord-cpp).
Implementation of [dscripka/openWakeWord](https://github.com/dscripka/openWakeWord).

### C Library headers
- [openwakeword.h](./include/openwakeword.h)

### Features
- A cpp runtime for openwakeword models
- Built in integration of portaudio to support different input devices (only tested in Linux though)
- A Unix style extensible CLI interface
- Also everything comes as a library to port this into something new/good..

### Regarding OpenWakeWord
- Feel free to checkout [dscripka/openWakeWord](https://github.com/dscripka/openWakeWord) for more details on,
  - what OpenWakeWord is (I am feeling bit lazy to write it here, rn)
  - how to train your custom wake word / hot word models
- Once you got your desired wakeword model, either [custom trained](https://github.com/dscripka/openWakeWord?tab=readme-ov-file#training-new-models) or [pretrained](https://github.com/dscripka/openWakeWord?tab=readme-ov-file#pre-trained-models), things are simple. point it to the lib/cli and let it do the rest.
- For an abundant(not thaat abundant) list of pretrained models: visit [this awesome repo](https://github.com/fwartner/home-assistant-wakewords-collection)

### CLI
```sh
usage: ./build/openwakeword [options]

Detect wake-word invocations from audio input. By default, audio is read from stdin. Use '-f' to read from a raw file or '-d' to read from the system's default microphone.

options:
   -h        --help                  show this message and exit
   -m  FILE  --model          FILE   path to wake word model (repeat for multiple models)
   -d        --mic                   read from default microphone of system 
   -f  FILE  --file           FILE   path to raw pcm data file 
             --list-mics             list all detected audio input devices 
   -i  ID    --device-id      ID     input device id to be used. if not provided default device will be used. 
   -e  CMD   --exec           CMD    command to be executed on detection time.
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
# ---------------- METHOD 1: Read from system microphone
# To read from default microphone of your system (detected by portaudio)
openwakeword --model misc/sheila_v2.onnx -d
# to get list of all the detected input devices
openwakeword --list-mics
# then specify the microphone device id to read from specific mic 
openwakeword --model misc/sheila_v2.onnx -d -i 21   # where 21 is the device ID

# or execute some custom commands on detection
openwakeword --model misc/sheila_v2.onnx -d -e "echo 'I aint Jarvis'"

# ---------------- METHOD 2: Read from STDIN
# Accepting it if incase you were not able to compile portaudio successfully for your device,
# read from ALSA input (for Linux Ofc)
arecord -r 16000 -c 1 -f S16_LE -t raw - | openwakeword --model misc/sheila_v2.onnx
# or using pipwire
pw-cat --record --format s16 --channels 1 --rate 16000 - | ./build/openwakeword --model misc/sheila_v2.onnx
# I mean feel free to use any feeder
# or exececute a command on detection time instead
arecord -r 16000 -c 1 -f S16_LE -t raw - | openwakeword --model misc/sheila_v2.onnx --silent -e "echo 'I aint Jarvis'"
```

### Build
- Make sure PulseAudio or your audio backend of choice is detected by the cmake build system on build time. (look for the configuration logs). Otherwise you know, the audio input may not work..
```sh
# use the make file for orchestration & cmake for compilation

# to get all targets
make

# build using 
make build

# generate prod build
make package
```

### Input Audio Specs (fmt: PCM)
* sample rate: 16000 Hz  (16kHz)
* bit depth: 16-bit 
* channels: 1
* minimal frame length: 80 ms
* bit format: s16_le
