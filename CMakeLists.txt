cmake_minimum_required(VERSION 3.20)
project(OpenWakeWordGstCpp VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

include_directories(include)
include_directories(libs/onnx/include)
link_directories(libs/onnx/lib)

# Port audio things
set(PORTAUDIO_BUILD_SHARED OFF)  # static build
set(PORTAUDIO_BUILD_TESTS OFF)
add_subdirectory(libs/portaudio)
# disable build time warning for portaudio (there exists some deprication warnings)
if(NOT MSVC)
    target_compile_options(portaudio PRIVATE -w)
else()
    target_compile_options(portaudio PRIVATE /w)
endif()




# --- Executable target ---
file(GLOB SRC_FILES "src/*.cpp")
add_executable(openwakeword ${SRC_FILES})
# Ensuring runtime can find shared libraries
set_target_properties(openwakeword PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
)

# --- Shared library target (C-compatible) ---
add_library(libopenwakeword SHARED ${SRC_FILES})
set_target_properties(libopenwakeword PROPERTIES OUTPUT_NAME "openwakeword")

# linking
target_link_libraries(openwakeword PRIVATE onnxruntime Threads::Threads)
target_link_libraries(openwakeword PRIVATE portaudio)
target_link_libraries(libopenwakeword PRIVATE onnxruntime Threads::Threads)
target_link_libraries(libopenwakeword PRIVATE portaudio)

target_include_directories(openwakeword PRIVATE libs/portaudio/include)
target_include_directories(libopenwakeword PRIVATE libs/portaudio/include)

install(TARGETS openwakeword RUNTIME DESTINATION bin)
install(TARGETS libopenwakeword LIBRARY DESTINATION lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/libs/onnx/lib/ DESTINATION lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/models/ DESTINATION models)
install(FILES ${CMAKE_SOURCE_DIR}/LICENSE DESTINATION .)

if(NOT MSVC)
    target_compile_options(openwakeword PRIVATE
        $<$<CONFIG:Debug>:-Wall -Wextra -Wpedantic -Werror>
    )
    target_compile_options(libopenwakeword PRIVATE
        $<$<CONFIG:Debug>:-Wall -Wextra -Wpedantic -Werror>
    )
endif()
