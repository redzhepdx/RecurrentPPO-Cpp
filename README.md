# RecurrentPPO-Cpp

<div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">

  <video width="320" controls style="max-width: 45%;">
    <source src="videos/VizDoom-Basic.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video width="320" controls style="max-width: 45%;">
    <source src="videos/VizDoom-Corridor.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

</div>

Recurrent PPO to solve Doom purely in CPP (Yuck). This is self-learning project.
I feel like it will be messy but it is what it is I guess!. 
The sources I followed:

1. Cleanrl : https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_lstm.py#L310
2. StableBaselines3 : https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
3. and SKRL for guidance and pseudo-code : https://skrl.readthedocs.io/en/latest/api/agents/ppo.html


##### Dependencies

## 1. General dependencies
```bash
sudo apt-get install clang clang++ gcc g++ cmake boost sdl2 openal-soft
```

or 

```bash
brew install clang clang++ gcc g++ cmake boost sdl2 openal-soft
```

### 2. `Libtorch 2.9.1`

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

## 3. Build the entire repo

`DO NOT FORGET TO RUN THIS FROM THE MAIN FOLDER OF THE PROJECT.`
```bash
mkdir build && cd build

cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . -v
```

## 4. Run training

Maybe you should modify the `src/main.cpp` if you want to change the environment. Search for `train_ppo_run`/`train_ppo_rnn_run` function. Similar to StableBaselines.

```
./recurrent_ppo_cpp --operation train
```

or

```
./recurrent_ppo_cpp --operation train_rnn
```

It takes around 400-2000 episodes depending on the scenario with vanilla PPO (PPO-RNN 20-50 updates). Start with simple.wad. It is default.

## 4. Evaluate the trained agent

#### IMPORTANT : If you changed something in `train_ppo_run`/`train_ppo_rnn_run` function related to the agent or environment, do the same changes in `play_doom_ai_run`/`play_doom_ai_rnn_run` otherwise you will use apples to squeeze orange juice and you will be very very sad.

```
./recurrent_ppo_cpp --operation play_ai
```

or

```
./recurrent_ppo_cpp --operation play_ai_rnn
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