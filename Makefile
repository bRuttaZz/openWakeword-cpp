
OS ?= linux
ARCH ?= x64

ONNX_VERSION ?= 1.23.1

BUILD_TYPE ?= Release
PACKAGE_DIR ?= dist

help:	## Show all Makefile targets.
	@grep -E '^[a-zA-Z_\/-]+:.*?## .*$$' Makefile | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

libs/onnx:
	@echo "\nEnsuring onnx libraries.."
	mkdir -p libs/onnx
	wget -q -O libs/onnx/onnxruntime.tgz \
	    https://github.com/microsoft/onnxruntime/releases/download/v$(ONNX_VERSION)/onnxruntime-$(OS)-$(ARCH)-$(ONNX_VERSION).tgz
	tar xvzf libs/onnx/onnxruntime.tgz -C libs/onnx/ --strip-components=1

libs/gst:
	@pkg-config --cflags --libs gstreamer-1.0 1>/dev/null 2>/dev/null || { \
	    echo "\nGstreamer Not found! Make sure Gstreamer libraries are available" 1>&2 && \
		exit 1; \
	}

models:
	@echo "\nDownloading onnx models.."
	mkdir -p models
	@echo "\nDownloading models/meslspectogram.onnx"
	wget -q -O models/melspectrogram.onnx https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx
	@echo "\nDownloading models/meslspectogram.onnx"
	wget -q -O models/embedding_model.onnx https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx
	@echo "Downloaded all onnx models.."

setup: libs/onnx libs/gst models	## Setup libraries
	@echo "\nConfigure.."
	cmake -S . -B build

build: setup 	## Build
	@echo "Creating $(BUILD_TYPE) build.."
	@mkdir build 2>/dev/null || true
	@cd build; cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) ..;
	@cd build; cmake --build .;

package: build ## Create package
	@echo "Creating package: $(PACKAGE_DIR) .."
	cmake --install ./build --prefix $(PACKAGE_DIR)

clean: ## Clean all build deps and builds
	$(RM) -rf build
	$(RM) -rf $(PACKAGE_DIR)

clean-all: clean ## Clean all downloaded assets assets
	$(RM) -rf libs
	$(RM) -rf models

.PHONY: setup package help
