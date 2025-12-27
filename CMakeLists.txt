cmake_minimum_required(VERSION 3.12)
project(recurrent_ppo_cpp)

# ===== TARGET SETTINGS =====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OSX_ARCHITECTURES "arm64") # Force ARM64 on Apple Silicon

# ======================
# LibTorch (PyTorch C++ API)
# ======================
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")
find_package(Torch REQUIRED)

# ======================
# Sources
# ======================
file(GLOB_RECURSE SOURCES "src/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "main.cpp")

# ======================
# Main executable
# ======================
add_executable(recurrent_ppo_cpp
    src/main.cpp
    ${SOURCES}
)

# ======================
# INCLUDE DIRECTORIES
# ======================
target_include_directories(recurrent_ppo_cpp PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# ======================
# BOOST (thread, system, filesystem, chrono)
# ======================

# Tell CMake to use the legacy FindBoost module
set(BOOST_ROOT "/opt/homebrew/opt/boost")
set(BOOST_INCLUDEDIR "/opt/homebrew/opt/boost/include")
set(BOOST_LIBRARYDIR "/opt/homebrew/opt/boost/lib")

set(Boost_NO_BOOST_CMAKE ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_STATIC_RUNTIME OFF)

# Add Homebrew include path so legacy FindBoost can find headers
set(CMAKE_INCLUDE_PATH "/opt/homebrew/include" ${CMAKE_INCLUDE_PATH})
set(CMAKE_LIBRARY_PATH "/opt/homebrew/opt/boost/lib" ${CMAKE_LIBRARY_PATH})

# Add Homebrew libomp rpath (Apple Silicon)
set(LIBOMP_PATH "/opt/homebrew/opt/libomp/lib")

if(APPLE)
    set(CMAKE_INSTALL_RPATH "${LIBOMP_PATH}")
    set(CMAKE_BUILD_RPATH "${LIBOMP_PATH}")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
endif()

find_package(Boost REQUIRED COMPONENTS
    thread

    # system
    filesystem
    chrono
)

if(Boost_FOUND)
    message(STATUS "Boost include dirs: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost libraries: ${Boost_LIBRARIES}")
    target_include_directories(recurrent_ppo_cpp PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(recurrent_ppo_cpp PRIVATE ${Boost_LIBRARIES})
endif()

# ======================
# Thread support (std/pthreads)
# ======================
find_package(Threads REQUIRED)
target_link_libraries(recurrent_ppo_cpp PRIVATE Threads::Threads)

# ======================
# ViZDoom C++ Library
# ======================
set(VIZDOOM_BUILD_DIR "${CMAKE_SOURCE_DIR}/ViZDoom/build/bin")
set(VIZDOOM_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/ViZDoom/include")

target_include_directories(recurrent_ppo_cpp PRIVATE "${VIZDOOM_INCLUDE_DIR}")

if(EXISTS "${VIZDOOM_BUILD_DIR}/libvizdoom.a")
    message(STATUS "Linking ViZDoom static library")
    target_link_libraries(recurrent_ppo_cpp PRIVATE "${VIZDOOM_BUILD_DIR}/libvizdoom.a")
elseif(EXISTS "${VIZDOOM_BUILD_DIR}/libvizdoom.dylib")
    message(STATUS "Linking ViZDoom dynamic library")
    target_link_libraries(recurrent_ppo_cpp PRIVATE "${VIZDOOM_BUILD_DIR}/libvizdoom.dylib")
else()
    message(FATAL_ERROR "ViZDoom library not found in: ${VIZDOOM_BUILD_DIR}")
endif()

# ======================
# SDL2 (graphics / input)
# ======================
find_package(SDL2 REQUIRED)

if(SDL2_FOUND)
    message(STATUS "SDL2 include dirs: ${SDL2_INCLUDE_DIRS}")
    target_include_directories(recurrent_ppo_cpp PRIVATE ${SDL2_INCLUDE_DIRS})
    target_link_libraries(recurrent_ppo_cpp PRIVATE SDL2::SDL2)
endif()

# ======================
# JPEG (image support)
# ======================
set(JPEG_LIB "/opt/homebrew/lib/libjpeg.dylib")

if(EXISTS "${JPEG_LIB}")
    target_include_directories(recurrent_ppo_cpp PRIVATE /opt/homebrew/include)
    target_link_libraries(recurrent_ppo_cpp PRIVATE "${JPEG_LIB}")
else()
    message(WARNING "libjpeg not found at /opt/homebrew/lib/libjpeg.dylib")
endif()

# ======================
# mpg123 (audio decoding support)
# ======================
set(MPG123_LIB "/opt/homebrew/lib/libmpg123.dylib")

if(EXISTS "${MPG123_LIB}")
    target_link_libraries(recurrent_ppo_cpp PRIVATE "${MPG123_LIB}")
endif()

# ======================
# libsndfile (audio file support)
# ======================
set(SNDFILE_LIB "/opt/homebrew/lib/libsndfile.dylib")

if(EXISTS "${SNDFILE_LIB}")
    target_link_libraries(recurrent_ppo_cpp PRIVATE "${SNDFILE_LIB}")
endif()

# ======================
# OpenAL (optional audio)
# ======================
find_library(OPENAL_LIB OpenAL)

if(OPENAL_LIB)
    target_link_libraries(recurrent_ppo_cpp PRIVATE "${OPENAL_LIB}")
endif()

# ======================
# LibTorch linking
# ======================
target_link_libraries(recurrent_ppo_cpp PRIVATE "${TORCH_LIBRARIES}")
