#include <iostream>
#include <memory>
#include <tuple>
#include <unistd.h>
#include <unordered_map>

#include <torch/torch.h>

#include "agent/policy.hpp"
#include "agent/value.hpp"
#include "buffer.hpp"
#include "envs.hpp"
#include "ppo.hpp"
#include "ppo_rnn.hpp"

void play_doom_run()
{
    // Randomly plays the game
    int64_t stack_frames   = 3;
    int64_t skip_frames    = 4;
    int64_t episode_length = 300;
    bool visualize         = false;
    DoomENV env(visualize, stack_frames, episode_length, skip_frames);
    std::cout << "Environment initialized." << std::endl;
    env.play(true);
}

void train_ppo_run()
{
    // Train Vanilla PPO on the given environment
    torch::manual_seed(1337);
    std::srand(1337);

    int64_t stack_frames   = 1;     // Store N frames
    int64_t skip_frames    = 4;     // Apply actions N times
    int64_t episode_length = 300;   // 300 for basic, 500 for deadly corridor
    bool visualize         = false; // Don't render
    bool combination       = true;  // Action combinations

    DoomENV env(visualize, stack_frames, episode_length, skip_frames, combination);

    // Check the struct to learn about the parameters
    PPOParams params;
    params.total_time_steps     = 10000;
    params.num_steps            = 2048;
    params.ppo_epochs           = 10;
    params.num_minibatches      = 32;
    params.learning_rate_policy = 3e-4;
    params.learning_rate_critic = 1e-5;
    params.ent_coef             = 0.0;
    params.clip_epsilon         = 0.2;
    params.clip_vloss           = false;
    params.anneal_lr            = true;
    params.reward_sharper       = 0.01;
    params.max_grad_norm        = 0.5;
    params.grad_acc_steps       = 4;
    bool separate_nets          = true;

    std::unique_ptr<PPO> ppo = std::make_unique<PPO>(env.get_observation_size(), env.get_action_size(), params, separate_nets);
    ppo->train(env, 1);
}

void play_doom_ai_run()
{
    // Play with Vanilla PPO, if you already have a trained agent
    // It will automatically load the last checkpoint (the best) from the "artifacts" folder
    torch::manual_seed(1337);
    std::srand(1337);

    int64_t stack_frames   = 1;
    int64_t skip_frames    = 4;
    int64_t episode_length = 300;
    bool visualize         = true;
    bool combination       = true;

    std::unique_ptr<DoomENV> env = std::make_unique<DoomENV>(visualize, stack_frames, episode_length, skip_frames, combination);
    PPOParams params;
    params.total_time_steps     = 100;
    params.num_steps            = 128;
    params.ppo_epochs           = 4;
    params.num_minibatches      = 8;
    params.learning_rate_policy = 1e-4;
    params.learning_rate_critic = 1e-5;
    bool separate_nets          = true;

    std::unique_ptr<PPO> ppo = std::make_unique<PPO>(env->get_observation_size(), env->get_action_size(), params, separate_nets);

    double total_return = 0.0;
    double total_win    = 0.0;

    std::cout << "Playing with trained agent..." << std::endl;
    for (int i = 0; i < 100; ++i) {
        double reward_sum     = 0;
        auto state            = env->reset();
        bool done_            = false;
        size_t episode_length = 0;
        while (!done_) {
            auto action                           = ppo->act(state);
            auto [next_state, reward, done, _, _] = env->step(action.squeeze(0));

            state = next_state;
            reward_sum += reward;
            done_ = done;
            episode_length++;

            usleep(100000);
        }

        std::cout << "Episode Length : " << episode_length << std::endl;
        std::cout << "Total Reward: " << reward_sum << std::endl;

        auto win = env->last_total_reward() > 0 ? 1.0 : 0.0;
        total_win += win;
        total_return += reward_sum;
        std::cout << "Total Win: " << total_win << std::endl;
    }
    std::cout << "AVG Win : " << total_win / 100.0 << std::endl;
    std::cout << "Win Count : " << total_win << std::endl;
    std::cout << "Lose Count : " << 100.0 - total_win << std::endl;
    std::cout << "AVG Episode Reward: " << total_return / 100 << std::endl;
}

void train_ppo_rnn_run()
{

    // Train Recurrent PPO on the given environment
    torch::manual_seed(1337);
    std::srand(1337);

    int64_t stack_frames   = 1;     // Store N frames
    int64_t skip_frames    = 4;     // Apply actions N times
    int64_t episode_length = 500;   // 300 for basic, 500 for deadly corridor
    bool visualize         = false; // Don't render
    bool combination       = true;  // Action combinations

    DoomENV env(visualize, stack_frames, episode_length, skip_frames, combination);

    // Check the struct to learn about the parameters
    PPO_RNN_PARAMS params;
    params.total_time_steps     = 10000 * 2048;
    params.num_steps            = 2048;
    params.ppo_epochs           = 10;
    params.num_minibatches      = 1;
    params.learning_rate_policy = 3e-4;
    params.learning_rate_critic = 1e-5;
    params.ent_coef             = 0.01;
    params.clip_epsilon         = 0.2;
    params.clip_vloss           = false;
    params.anneal_lr            = true;
    params.reward_sharper       = 0.01;
    params.max_grad_norm        = 0.5;
    params.grad_acc_steps       = 4;
    bool separate_nets          = true;

    std::unique_ptr<PPO_RNN> ppo = std::make_unique<PPO_RNN>(env.get_observation_size(), env.get_action_size(), params, separate_nets);
    ppo->train(env, 1);
}

void play_doom_ai_rnn_run()
{

    // Play with Vanilla PPO, if you already have a trained agent
    // It will automatically load the last checkpoint (the best) from the "artifacts_rnn" folder
    torch::manual_seed(1337);
    std::srand(1337);

    int64_t stack_frames   = 1;
    int64_t skip_frames    = 4;
    int64_t episode_length = 500;
    bool visualize         = true;
    bool combination       = true;

    std::unique_ptr<DoomENV> env = std::make_unique<DoomENV>(visualize, stack_frames, episode_length, skip_frames, combination);
    PPO_RNN_PARAMS params;
    params.total_time_steps     = 100;
    params.num_steps            = 128;
    params.ppo_epochs           = 4;
    params.num_minibatches      = 8;
    params.learning_rate_policy = 1e-4;
    params.learning_rate_critic = 1e-5;
    bool separate_nets          = true;

    std::unique_ptr<PPO_RNN> ppo = std::make_unique<PPO_RNN>(env->get_observation_size(), env->get_action_size(), params, separate_nets);

    double total_return = 0.0;
    double total_win    = 0.0;

    std::cout << "Playing with trained agent..." << std::endl;
    for (int i = 0; i < 100; ++i) {
        double reward_sum     = 0;
        auto state            = env->reset();
        bool done_            = false;
        size_t episode_length = 0;
        while (!done_) {
            auto action                           = ppo->act(state);
            auto [next_state, reward, done, _, _] = env->step(action.squeeze(0));

            state = next_state;
            reward_sum += reward;
            done_ = done;
            episode_length++;

            usleep(100000);
        }

        std::cout << "Episode Length : " << episode_length << std::endl;
        std::cout << "Total Reward: " << reward_sum << std::endl;

        auto win = env->last_total_reward() > 0 ? 1.0 : 0.0;
        total_win += win;
        total_return += reward_sum;
        std::cout << "Total Win: " << total_win << std::endl;
    }
    std::cout << "AVG Win : " << total_win / 100.0 << std::endl;
    std::cout << "Win Count : " << total_win << std::endl;
    std::cout << "Lose Count : " << 100.0 - total_win << std::endl;
    std::cout << "AVG Episode Reward: " << total_return / 100 << std::endl;
}

std::unordered_map<std::string, std::string> parse_args(int argc, char* argv[])
{
    std::unordered_map<std::string, std::string> args;
    for (size_t arg_index = 0; arg_index < argc; ++arg_index) {
        std::string arg = argv[arg_index];
        if (arg.rfind("--", 0) == 0) {
            args[arg] = argv[arg_index + 1];
        }
    }

    return args;
}

int main(int argc, char* argv[])
{
    auto args = parse_args(argc, argv);

    if (args["--operation"].compare("train") == 0) {
        train_ppo_run();
    } else if (args["--operation"].compare("play_ai") == 0) {
        play_doom_ai_run();
    } else if (args["--operation"].compare("train_rnn") == 0) {
        train_ppo_rnn_run();
    } else if (args["--operation"].compare("play_ai_rnn") == 0) {
        play_doom_ai_rnn_run();
    } else {
        play_doom_run();
    }

    return 0;
}