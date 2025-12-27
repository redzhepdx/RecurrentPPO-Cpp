#include <iostream>
#include <torch/torch.h>

#include "agent/policy.hpp"
#include "agent/value.hpp"
#include "buffer.hpp" /
#include "envs.hpp"
#include "ppo.hpp"

int main()
{
    DoomENV env(true);
    std::cout << "Environment initialized." << std::endl;
    env.play(true);
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