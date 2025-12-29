#include <iostream>
#include <torch/torch.h>

#include "agent/policy.hpp"
#include "agent/value.hpp"
#include "buffer.hpp" /
#include "envs.hpp"
#include "ppo.hpp"

void play_doom_run()
{
    DoomENV env(true);
    std::cout << "Environment initialized." << std::endl;
    env.play(true);
}

void train_ppo_run()
{
    torch::manual_seed(42);
    std::srand(42);

    DoomENV env(false);
    PPOParams params;
    params.total_time_steps  = 100;
    params.num_steps         = 1024;
    params.ppo_epochs        = 4;
    params.num_minibatches   = 8;
    params.learning_rate     = 1e-4;
    std::unique_ptr<PPO> ppo = std::make_unique<PPO>(env.get_observation_size(), env.get_action_size(), params);
    ppo->train(env, 1);
}

int main()
{
    // DoomENV env(true);
    // std::cout << "Environment initialized." << std::endl;
    // env.play(true);
    // play_doom_run();
    train_ppo_run();

    // Create environment
    // DoomENV env;
    // Create PPO Trainer
    // PPO ppo_trainer(env.get_observation_size(), env.get_action_size());
    // Train PPO
    // ppo_trainer.train(env, 1000);
    // Play with trained agent
    // while (true) {
    //   auto state = env.reset();
    //   bool done = false;
    //   while (!done) {
    //     auto action = ppo_trainer.act(state);
    //     auto [next_state, reward, done] = env.step(action);
    //     state = next_state;
    //   }

    return 0;
}