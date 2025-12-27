# RecurrentPPO-Cpp
Recurrent PPO to solve Doom purely in CPP (Yuck). This is self-learning project.
I feel like it will be messy but it is what it is I guess!.


##### Dependencies

1. General dependencies
```bash
sudo apt-get install clang clang++ gcc g++ cmake boost sdl2 openal-soft
```

or 

```bash
brew install clang clang++ gcc g++ cmake boost sdl2 openal-soft
```

1. `Libtorch 2.9.1`

Download and unzip in this project.

LINK FOR MACOS: https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.9.1.zip

All the libtorch dists : https://download.pytorch.org/libtorch/cpu/?utm_source=chatgpt.com

2. `ViZDoom` (Good luck with dependencies on MACOS)

```bash
git clone https://github.com/Farama-Foundation/ViZDoom.git
cd ViZDoom
mkdir build && cd build

cmake .. \                                               
 -DCMAKE_BUILD_TYPE=Release \
 -DBUILD_ENGINE=ON \
 -DBUILD_PYTHON=OFF \
 -DCMAKE_OSX_ARCHITECTURES=arm64
  
make -j$(sysctl -n hw.ncpu)

cd ../..
```

3. Build the entire repo

`DO NOT FORGET TO RUN THIS FROM THE MAIN FOLDER OF THE PROJECT.`
```bash
mkdir build && cd build

cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . -v
```

#### VSCode Setup 

`.vscode/c_cpp_properties.json`

```
{
    "configurations": [
        {
            "name": "Mac",
            "includePath": [
                "${workspaceFolder}/**",
                "/Library/Developer/CommandLineTools/usr/include/c++/v1", // C++ standard headers
                "/usr/local/include",
                "/usr/include",
                "CUSTOM_PATH_TO_PARENT/RecurrentPPO-Cpp/libtorch/include"
            ],
            "compilerPath": "/usr/bin/clang++", // Use Clang
            "cStandard": "c17",
            "cppStandard": "c++17",
            "intelliSenseMode": "macos-clang-arm64"
        }
    ],
    "version": 4
}
```



##IMPORTANT! 

You should install only arm64 version of everything for macos. Also deal with the cmake.
I had to use CMake because libtorch and vizdoom is crying for cmake. 
If I could avoid, I would but it would just make the process 10x longer.
I am not a cmake master and I despise it to be very frank.
Maybe one day I will port everything to NOB and be happy.