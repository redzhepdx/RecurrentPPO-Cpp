#pragma once

#include <memory>

#include "ds.hpp"

#include <ViZDoom.h>
#include <torch/torch.h>

torch::Tensor bits_from_scalar(const torch::Tensor& scalar_tensor, int64_t num_bits)
{
    int64_t value = scalar_tensor.item<int64_t>();

    std::vector<int64_t> bits;
    bits.reserve(num_bits);

    for (int64_t i = 0; i < num_bits; ++i) {
        // Extract i-th bit
        int64_t bit = (value >> i) & 1;
        bits.push_back(bit);
    }
    return torch::tensor(bits, torch::kFloat32);
}

// Abstract Environment class
class Env {
  public:
    virtual torch::Tensor reset()                                                                 = 0;
    virtual std::tuple<torch::Tensor, double, bool, bool, bool> step(const torch::Tensor& action) = 0;
    virtual std::tuple<size_t, size_t, size_t> get_observation_size() const                       = 0;
    virtual size_t get_action_size() const                                                        = 0;
    virtual ~Env()                                                                                = default;
    virtual double last_total_reward() const                                                      = 0;
    virtual double last_reward() const                                                            = 0;
};

class DoomENV : public Env {
  private:
    std::unique_ptr<vizdoom::DoomGame> game_;
    std::unique_ptr<CircularTensorBuffer> states_queue_;

    int64_t state_stack_size_ = 4;
    int64_t skip_frames_      = 4;
    bool combination          = false;
    int left_decision = 0, right_decision = 0, decision_count = 0;

  public:
    DoomENV(bool visible = false, size_t state_stack_size = 4, size_t episode_time_out = 200, int64_t skip_frames = 4, bool combination = false)
        : state_stack_size_(state_stack_size), skip_frames_(skip_frames), combination(combination)
    {
        game_ = std::make_unique<vizdoom::DoomGame>();
        // Sets path to vizdoom engine executive which will be spawned as a separate
        // process.Default is "./vizdoom".
        game_->setViZDoomPath("../ViZDoom/build/bin/vizdoom.app/Contents/MacOS/vizdoom");

        // Sets path to doom2 iwad resource file which contains the actual doom
        // game->Default is "./doom2.wad".
        game_->setDoomGamePath("../ViZDoom/src/freedoom2.wad");
        // game_->setDoomGamePath("../../bin/doom2.wad");
        // Not provided with environment due to licences
        // .

        game_->loadConfig("../ViZDoom/scenarios/basic.cfg");

        //   Sets path to additional resources iwad file which is basically your scenario iwad.
        //   If not specified default doom2 maps will be used and it's pretty much useless...
        //  unless
        //   you want to play doom.
        game_->setDoomScenarioPath("../ViZDoom/scenarios/basic.wad");

        // Set map to start(scenario.wad files can contain many maps).
        game_->setDoomMap("map01");

        // Sets resolution.Default is 320X240
        game_->setScreenResolution(vizdoom::RES_160X120);

        // Sets the screen buffer format.Not used here but now you can change it.Default is
        // CRCGCB.
        game_->setScreenFormat(vizdoom::GRAY8);

        game_->setWindowVisible(visible);

        game_->addAvailableButton(vizdoom::MOVE_LEFT);
        game_->addAvailableButton(vizdoom::MOVE_RIGHT);
        game_->addAvailableButton(vizdoom::ATTACK);

        // game_->addAvailableButton(vizdoom::MOVE_LEFT_RIGHT_DELTA, 1);

        // Adds game variables that will be included in state.
        game_->addAvailableGameVariable(vizdoom::AMMO2); // Causes episodes to finish after 200 tics (actions)
        game_->setEpisodeTimeout(episode_time_out);

        // Makes episodes start after 10 tics (~after raising the weapon)
        game_->setEpisodeStartTime(10);

        // This is important if you are doing multiple step frame skipping
        game_->setLivingReward(-1.0 / skip_frames);

        states_queue_ = std::make_unique<CircularTensorBuffer>(
            1, game_->getScreenHeight() / 2, game_->getScreenWidth() / 2, torch::Device("cpu"), state_stack_size_);

        game_->init();
    };

    torch::Tensor reset() override
    {
        game_->newEpisode();
        vizdoom::GameStatePtr state = game_->getState();
        torch::Tensor screen_tensor =
            torch::from_blob(state->screenBuffer->data(), {1, game_->getScreenHeight(), game_->getScreenWidth(), 1}, torch::kUInt8)
                .permute({0, 3, 1, 2})
                .to(torch::kFloat32) /
            255.0;

        // Resize 120x160 -> 60x80
        screen_tensor = torch::nn::functional::interpolate(screen_tensor,
                                                           torch::nn::functional::InterpolateFuncOptions()
                                                               .size(std::vector<int64_t>{screen_tensor.size(2) / 2, screen_tensor.size(3) / 2})
                                                               .mode(torch::kBilinear)
                                                               .align_corners(false));

        states_queue_->reset();
        states_queue_->push_front(screen_tensor[0]);

        return states_queue_->view().unsqueeze(0);
    }

    void collect_decision_metrics(const torch::Tensor& action)
    {
        decision_count++;

        if (action.size(-1) == 1) {
            if (action.item<float>() > 0) {
                left_decision++;
            } else {
                right_decision++;
            }
        } else {
            if (action[0].item<float>() == 1) {
                left_decision++;
            }
            if (action[1].item<float>() == 1) {
                right_decision++;
            }
        }

        if (decision_count % 1000 == 0) {
            std::cout << "left_frac=" << (double)left_decision / decision_count << std::endl;
            std::cout << "right_frac=" << (double)right_decision / decision_count << std::endl;
            left_decision  = 0;
            right_decision = 0;
            decision_count = 0;
        }
    }

    torch::Tensor preprocess_action(const torch::Tensor& action)
    {
        torch::Tensor action_to_register;
        if (action.size(-1) == 1) {
            if (combination) {
                // Example 1 -> (0, 0, 1), 5 ->(1, 0, 1) etc
                action_to_register = bits_from_scalar(action, (int64_t)game_->getAvailableButtonsSize());
            } else {
                // One hot encoding
                action_to_register = torch::zeros({(int64_t)game_->getAvailableButtonsSize()}, torch::kFloat);
                action_to_register.scatter_(0, action, 1.0);
            }

        } else {
            // Direct prediction or sigmoid output
            action_to_register = action;
        }
        return action_to_register;
    }

    std::tuple<torch::Tensor, double, bool, bool, bool> step(const torch::Tensor& action) override
    {
        torch::Tensor action_to_register = preprocess_action(action);

        collect_decision_metrics(action_to_register);

        // Clamp the action values to valid range
        std::vector<double> action_vector(action_to_register.data_ptr<float>(), action_to_register.data_ptr<float>() + action_to_register.numel());

        double reward = game_->makeAction(action_vector, skip_frames_);

        bool done = game_->isEpisodeFinished();

        bool terminated = game_->isEpisodeTimeoutReached(); // We don't need it but just in case for the future
        bool truncated  = !terminated && done;

        // Get the current state if not done
        vizdoom::GameStatePtr state = nullptr;
        if (!done) {
            state = game_->getState();
        }

        // Create observation tensor
        torch::Tensor obs_tensor;
        if (state) {
            obs_tensor = torch::from_blob(state->screenBuffer->data(), {1, game_->getScreenHeight(), game_->getScreenWidth(), 1}, torch::kUInt8)
                             .permute({0, 3, 1, 2})
                             .to(torch::kFloat32) /
                         255.0;

            // Resize 120x160 -> 60x80
            obs_tensor = torch::nn::functional::interpolate(obs_tensor,
                                                            torch::nn::functional::InterpolateFuncOptions()
                                                                .size(std::vector<int64_t>{obs_tensor.size(2) / 2, obs_tensor.size(3) / 2})
                                                                .mode(torch::kBilinear)
                                                                .align_corners(false));

        } else {
            // If done, produce a zero state
            obs_tensor = torch::zeros({1, 1, game_->getScreenHeight() / 2, game_->getScreenWidth() / 2}, torch::kFloat32);
        }

        // Add it to the buffer to stack the frames
        states_queue_->push_front(obs_tensor[0]);

        // If the episode ended, reset the environment for next time
        if (done) {
            states_queue_->reset();
        }

        return {states_queue_->view().unsqueeze(0), reward, done, terminated, truncated};
    }

    std::tuple<size_t, size_t, size_t> get_observation_size() const override
    {
        size_t channels = state_stack_size_;
        size_t width    = game_->getScreenWidth() / 2;
        size_t height   = game_->getScreenHeight() / 2;
        return {channels, height, width};
    }

    size_t get_action_size() const override
    {
        if (combination) {
            return (size_t)std::powl(2, game_->getAvailableButtonsSize());
        } else {
            return (size_t)game_->getAvailableButtonsSize();
        }
    }
    void play(bool random_act = false)
    {
        for (int i = 0; i < 10; ++i) {

            std::cout << "Episode #" << i + 1 << "\n";
            game_->newEpisode();
            while (!game_->isEpisodeFinished()) {

                vizdoom::GameStatePtr state = game_->getState();

                std::vector<double> action(game_->getAvailableButtonsSize());
                // Set your action.

                if (random_act) {
                    for (size_t i = 0; i < action.size(); ++i) {
                        action[i] = rand() % 2;
                    }
                } else {
                    // Example:
                    // pressing the first button
                    action[0] = 1;
                }

                game_->makeAction(action, skip_frames_);

                if (random_act) {
                    std::cout << "Performing random action!" << action << std::endl;
                    game_->makeAction(action, skip_frames_);
                }

                if (game_->isPlayerDead()) {
                    // Check if player is dead
                    // Use this to respawn immediately after death, new state will be available.
                    game_->respawnPlayer();
                }

                std::cout << game_->getEpisodeTime() << " Frags: " << game_->getGameVariable(vizdoom::FRAGCOUNT) << std::endl;
            }
        }
        game_->close();
    }

    double last_reward() const override { return game_->getLastReward(); }
    double last_total_reward() const override { return game_->getTotalReward(); }

    ~DoomENV() override
    {
        std::cout << "Closing DoomENV..." << std::endl;
        game_->close();
    }
};