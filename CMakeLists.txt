cmake_minimum_required(VERSION 3.20)
project(OpenWakeWordGstCpp VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(include)
include_directories(libs/onnx/include)

link_directories(libs/onnx/lib)

find_package(Threads REQUIRED)

file(GLOB SRC_FILES "src/*.cpp")

add_executable(openwakeword ${SRC_FILES})

target_link_libraries(openwakeword PRIVATE onnxruntime Threads::Threads)

# Ensure runtime can find shared libraries
set_target_properties(openwakeword PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
)

install(TARGETS openwakeword RUNTIME DESTINATION bin)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/libs/onnx/lib/ DESTINATION lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/models/ DESTINATION models)
install(FILES ${CMAKE_SOURCE_DIR}/LICENSE DESTINATION .)
