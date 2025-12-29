#pragma once

#include <ViZDoom.h>
#include <memory>
#include <torch/torch.h>

// Abstract Environment class
class Env {
  public:
    virtual torch::Tensor reset()                                              = 0;
    virtual std::tuple<torch::Tensor, double, bool> step(torch::Tensor action) = 0;
    virtual std::tuple<size_t, size_t, size_t> get_observation_size() const    = 0;
    virtual size_t get_action_size() const                                     = 0;
    virtual ~Env()                                                             = default;
};

class DoomENV : public Env {
  private:
    std::unique_ptr<vizdoom::DoomGame> game_;
    vizdoom::GameStatePtr prev_state_;
    double prev_reward_;

  public:
    DoomENV(bool visible = false)
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
        game_->setScreenFormat(vizdoom::RGB24);

        game_->setWindowVisible(visible);

        game_->addAvailableButton(vizdoom::MOVE_LEFT);
        game_->addAvailableButton(vizdoom::MOVE_RIGHT);
        game_->addAvailableButton(vizdoom::ATTACK);

        // Adds game variables that will be included in state.
        game_->addAvailableGameVariable(vizdoom::AMMO2); // Causes episodes to finish after 200 tics (actions)
        game_->setEpisodeTimeout(200);

        // Makes episodes start after 10 tics (~after raising the weapon)
        game_->setEpisodeStartTime(10);

        game_->init();
    };

    torch::Tensor reset() override
    {
        // game_->close();
        game_->newEpisode();
        vizdoom::GameStatePtr state = game_->getState();
        torch::Tensor screen_tensor =
            torch::from_blob(state->screenBuffer->data(), {1, game_->getScreenHeight(), game_->getScreenHeight(), 3}, torch::kUInt8)
                .permute({0, 3, 1, 2})
                .to(torch::kFloat32);
        screen_tensor /= 255.0;
        return screen_tensor;
    }

    std::tuple<torch::Tensor, double, bool> step(torch::Tensor action) override
    {
        std::vector<double> action_vector(action.data_ptr<float>(), action.data_ptr<float>() + action.numel());

        double reward = game_->makeAction(action_vector);
        bool done     = game_->isEpisodeFinished();

        // Get the current state if not done
        vizdoom::GameStatePtr state = nullptr;
        if (!done) {
            state = game_->getState();
        }

        // Create observation tensor
        torch::Tensor obs_tensor;
        if (state) {
            obs_tensor = torch::from_blob(state->screenBuffer->data(), {1, game_->getScreenHeight(), game_->getScreenHeight(), 3}, torch::kUInt8)
                             .permute({0, 3, 1, 2})
                             .to(torch::kFloat32) /
                         255.0;
        } else {
            // If done, produce a zero state or your chosen terminal state representation
            obs_tensor = torch::zeros({1, 3, game_->getScreenHeight(), game_->getScreenHeight()}, torch::kFloat32);
        }

        // If the episode ended, reset the environment for next time
        if (done) {
            game_->newEpisode();
        }

        return {obs_tensor, reward, done};
    }

    std::tuple<size_t, size_t, size_t> get_observation_size() const override
    {
        size_t channels = game_->getScreenChannels();
        size_t width    = game_->getScreenWidth();
        size_t height   = game_->getScreenHeight();
        return {channels, width, height};
    }

    size_t get_action_size() const override { return (size_t)game_->getAvailableButtonsSize(); }

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
                    action[0] = 1;
                    // Example:
                    // pressing the first button
                }

                game_->makeAction(action);

                if (random_act) {
                    std::cout << "Performing random action!" << action << std::endl;
                    game_->makeAction(action);
                }

                if (game_->isPlayerDead()) {
                    // Check if player is dead
                    game_->respawnPlayer();
                    // Use this to respawn immediately after death, new state will be available.
                }

                std::cout << game_->getEpisodeTime() << " Frags: " << game_->getGameVariable(vizdoom::FRAGCOUNT) << std::endl;
            }
        }
        game_->close();
        // game_.release();
    }

    ~DoomENV() override
    {
        if (game_->isRunning()) {
            game_->close();
        }
    }
};