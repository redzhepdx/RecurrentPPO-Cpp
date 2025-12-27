#pragma once
#include "agent/policy.hpp"
#include "agent/value.hpp"
#include "buffer.hpp"
#include "envs.hpp"
#include <memory>
#include <torch/torch.h>
#include <tuple>

class PPO {
  private:
    size_t batch_size_      = 64;
    size_t mini_batch_size_ = 64;
    size_t ppo_epochs_      = 10;
    size_t buffer_capacity_ = 32;
    size_t horizon_         = 64;
    size_t num_minibatches  = 32;
    double gamma_           = 0.99; // the discount factor gamma
    double lambda_          = 0.95; // the lambda for the general advantage estimation
    double clip_epsilon_    = 0.2;
    double learning_rate_   = 3e-4;
    double ent_coef         = 0.0;
    double vf_coef          = 0.5;
    double max_grad_norm    = 0.5;
    double target_kl        = 0.5;
    bool anneal_lr_         = true;
    bool clip_vloss_        = true;

    std::tuple<size_t, size_t, size_t> obs_size_;

    CNNPolicyNetwork policy_net;
    CNNValueNetwork value_net;

    std::unique_ptr<torch::optim::Adam> policy_optimizer;
    std::unique_ptr<torch::optim::Adam> value_optimizer;

    std::unique_ptr<ReplayBuffer> buffer;

    torch::Tensor batch_indices_;

  public:
    PPO(std::tuple<size_t, size_t, size_t> obs_size,
        int action_size,
        size_t buffer_capacity = 32,
        size_t batch_size      = 64,
        size_t ppo_epochs      = 10,
        double gamma           = 0.99,
        double clip_epsilon    = 0.2,
        size_t horizon         = 64)
        : buffer_capacity_(buffer_capacity), batch_size_(batch_size), ppo_epochs_(ppo_epochs), gamma_(gamma), clip_epsilon_(clip_epsilon),
          horizon_(horizon), obs_size_(obs_size)
    {

        policy_net = CNNPolicyNetwork(obs_size, action_size);
        value_net  = CNNValueNetwork(obs_size);

        policy_optimizer = std::make_unique<torch::optim::Adam>(policy_net->parameters(), torch::optim::AdamOptions(learning_rate_));

        value_optimizer = std::make_unique<torch::optim::Adam>(value_net->parameters(), torch::optim::AdamOptions(learning_rate_));

        buffer = std::make_unique<ReplayBuffer>(buffer_capacity_);

        batch_indices_ = torch::arange((int64_t)batch_size_);
    };

    void store_trajectory(const TrajectoryBuffer& trajectory) { buffer->add_trajectory(trajectory); }

    void clear_buffer() { buffer->clear(); }

    std::tuple<double, double> calculate_clip_fraction(torch::Tensor& log_ratio, torch::Tensor& ratio)
    {
        torch::NoGradGuard no_grad;
        // auto old_approx_kl = (-log_ratio).mean();

        auto approx_kl = ((ratio - 1) - log_ratio).mean();

        auto clipped_mask = (ratio - 1.0).abs() > clip_epsilon_;

        auto clipped_float = clipped_mask.to(torch::kFloat);

        auto clip_frac_tensor = clipped_float.mean();

        double clip_frac = clip_frac_tensor.item<double>();

        return {clip_frac, approx_kl.item<double>()};
    }

    void optimize(const TrajectoryBuffer& tb, torch::Tensor& advantages, torch::Tensor& returns)
    {
        size_t channel, width, height;
        std::tie(channel, width, height) = obs_size_;
        // Perform PPO optimization step
        auto states      = tb.get_states().reshape((-1, channel, width, height));
        auto actions     = tb.get_actions().reshape(-1);
        auto rewards     = tb.get_rewards().reshape(-1);
        auto dones       = tb.get_dones().reshape(-1);
        auto next_states = tb.get_next_states().reshape((-1, channel, width, height));
        auto log_probs   = tb.get_log_probs().reshape(-1);
        auto values      = tb.get_values().reshape(-1);

        // Normalize advantages
        auto norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8);

        // TODO probably reshape the states

        std::vector<double> clip_fractions;
        auto b_indices = batch_indices_.clone();
        for (size_t epoch = 0; epoch < ppo_epochs_; ++epoch) {

            // Generate a random permutation of indices
            auto perm = torch::randperm(b_indices.size(0));

            // Index the tensor with the random indices
            b_indices = b_indices.index({perm});

            for (size_t start = 0; start < batch_size_; start += mini_batch_size_) {
                auto end                = start + mini_batch_size_;
                auto mini_batch_indices = b_indices.slice(0, start, end);

                auto mb_states     = states.index({mini_batch_indices});
                auto mb_action     = actions.index({mini_batch_indices});
                auto mb_values     = values.index({mini_batch_indices});
                auto mb_returns    = returns.index({mini_batch_indices});
                auto mb_advantages = norm_advantages.index({mini_batch_indices});
                auto mb_log_probs  = log_probs.index({mini_batch_indices});

                auto [new_action, new_log_prob, entropy] = policy_net->get_actions(mb_states, mb_action);
                auto new_value                           = value_net->get_value(mb_states);

                auto log_ratio = new_log_prob - mb_log_probs;
                auto ratio     = log_ratio.exp();

                auto [clip_frac, approx_kl] = calculate_clip_fraction(log_ratio, ratio);
                clip_fractions.push_back(clip_frac);

                // mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8);

                // Policy gradient loss
                auto policy_gradient_loss         = -mb_advantages * ratio;
                auto policy_gradient_loss_clipped = -mb_advantages * torch::clamp(ratio, 1.0 - clip_epsilon_, 1.0 + clip_epsilon_);
                auto pg_loss                      = torch::max(policy_gradient_loss, policy_gradient_loss_clipped).mean();

                // Value loss
                new_value = new_value.view(-1);
                torch::Tensor v_loss;
                if (clip_vloss_) {
                    auto v_loss_raw     = torch::pow(new_value - mb_returns, 2);
                    auto v_clipped      = mb_values + torch::clamp(new_value - mb_returns, -clip_epsilon_, clip_epsilon_);
                    auto v_loss_clipped = torch::pow(v_clipped - mb_returns, 2);
                    auto v_loss_max     = torch::max(v_loss_clipped, v_loss_raw);
                    v_loss              = 0.5 * v_loss_max.mean();
                } else {
                    auto v_loss_raw = torch::pow(new_value - mb_returns, 2);
                    v_loss          = 0.5 * v_loss_raw.mean();
                }

                auto entropy_loss = entropy.mean();

                auto loss_policy = pg_loss + entropy_loss * ent_coef;
                auto loss_value  = v_loss * vf_coef;
                auto total_loss  = loss_policy + loss_value;

                value_optimizer->zero_grad();
                policy_optimizer->zero_grad();

                total_loss.backward();

                torch::nn::utils::clip_grad_norm_(policy_net->parameters(), max_grad_norm);
                torch::nn::utils::clip_grad_norm_(value_net->parameters(), max_grad_norm);

                policy_optimizer->step();
                value_optimizer->step();

                if (approx_kl > target_kl) {
                    break;
                }
            }
        }

        // Publish some statistics
    }

    std::tuple<torch::Tensor, torch::Tensor> calculate_advantages_and_returns(const TrajectoryBuffer& tb)
    {
        torch::NoGradGuard no_grad;

        auto states      = tb.get_states();
        auto actions     = tb.get_actions();
        auto rewards     = tb.get_rewards();
        auto dones       = tb.get_dones();
        auto next_states = tb.get_next_states();
        auto log_probs   = tb.get_log_probs();
        auto values      = tb.get_values();

        auto next_values = value_net->get_value(next_states);
        auto advantages  = torch::zeros_like(next_values);

        torch::Tensor last_gae = torch::zeros_like(values[0]);
        torch::Tensor next_non_terminal;
        torch::Tensor curr_next_value;

        auto last_done       = dones[dones.size(0) - 1];
        auto last_next_value = next_states[next_values.size(0) - 1];
        for (int64_t t = horizon_ - 1; t >= 0; --t) {

            if (t == horizon_ - 1) {
                // Special scenario that I need to understand later
                next_non_terminal = 1.0 - last_done;
                curr_next_value   = last_next_value;
            } else {
                next_non_terminal = 1.0 - dones[t + 1];
                curr_next_value   = values[t + 1];
            }

            // r + gamma * v'[t] * done - v[t]
            // advantage[t] = delta + gamma * gae_lambda * done * last_gae
            auto delta    = rewards[t] + gamma_ * curr_next_value * next_non_terminal - values[t];
            advantages[t] = delta + gamma_ * lambda_ * next_non_terminal * last_gae;
            last_gae      = advantages[t];
        }

        auto returns = advantages + values;

        return {advantages, returns};
    }

    void collect(Env& env)
    {
        TrajectoryBuffer trajectory_buffer(horizon_);
        auto state = env.reset();
        for (size_t t = 0; t < horizon_; ++t) {
            auto [action, log_prob, entropy] = policy_net->get_actions(state.unsqueeze(0));
            auto [next_state, reward, done]  = env.step(action.squeeze(0));
            auto value                       = value_net->forward(state.unsqueeze(0)).flatten();
            auto reward_tensor               = torch::full({}, reward, state.options());
            auto done_tensor                 = torch::full({}, done ? 1.0 : 0.0, state.options());

            trajectory_buffer.add(state, action.squeeze(0), log_prob, reward_tensor, next_state, done_tensor, value);
        }

        buffer->add_trajectory(trajectory_buffer);
    }

    void anneal_lr(size_t episode, size_t max_episodes)
    {
        auto frac   = 1.0 - ((double)episode - 1.0) / max_episodes;
        auto new_lr = frac * learning_rate_;

        // Update optimizers with new learning rate
        for (auto* optimizer : std::array{policy_optimizer.get(), value_optimizer.get()}) {
            for (auto& group : optimizer->param_groups()) {
                auto& options = static_cast<torch::optim::AdamOptions&>(group.options());
                options.lr(new_lr);
            }
        }
    }

    torch::Tensor act(torch::Tensor& state)
    {
        // Get action from policy network
        auto [action, log_prob, entropy] = policy_net->get_actions(state.unsqueeze(0));
        return action.squeeze(0);
    }

    void train(Env& env, size_t max_episodes)
    {
        for (size_t episode = 0; episode < max_episodes; ++episode) {
            anneal_lr(episode, max_episodes);
            collect(env);

            auto trajectories = buffer->sample_tbs(1);
            for (const auto& tb_ref : trajectories) {
                const auto& tb             = tb_ref.get();
                auto [advantages, returns] = calculate_advantages_and_returns(tb);
                optimize(tb, advantages, returns);
            }
            TrajectoryBuffer tb = buffer->get_last(); // For now only last

            auto [advantages, returns] = calculate_advantages_and_returns(tb);
            optimize(tb, advantages, returns);
        }
    }

    ~PPO() = default;
};
