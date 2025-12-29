#pragma once
#include "agent/policy.hpp"
#include "agent/value.hpp"
#include "buffer.hpp"
#include "envs.hpp"
#include <filesystem>
#include <memory>
#include <torch/torch.h>
#include <tuple>

typedef struct {
    size_t total_time_steps = 1000000; // total timesteps of the experiments
    double learning_rate    = 3e-4;    // the learning rate of the optimizer
    size_t num_envs         = 1;       // the number of parallel game environments. 1 for now!
    size_t num_steps        = 2048;    // the number of steps to run in each environment per policy rollout
    bool anneal_lr          = true;    // Toggle learning rate annealing for policy and value networks
    double gamma            = 0.99;    // the discount factor gamma
    double gae_lambda       = 0.95;    // the lambda for the general advantage estimation
    size_t num_minibatches  = 32;      // the number of mini-batches
    size_t ppo_epochs       = 10;      // the K epochs to update the policy
    bool norm_adv           = true;    // Toggles advantages normalization
    double clip_epsilon     = 0.2;     // the surrogate clipping coefficient
    bool clip_vloss         = true;    // Toggles whether or not to use a clipped loss for the value function, as per the paper.
    double ent_coef         = 0.0;     // coefficient of the entropy
    double vf_coef          = 0.5;     // coefficient of the value function
    double max_grad_norm    = 0.5;     // the maximum norm for the gradient clipping
    double target_kl        = 0.02;    // the target KL divergence threshold

    // size_t horizon_        = 64;
} PPOParams;

class PPO {
  private:
    PPOParams params_;

    size_t batch_size_      = 0; // batch_size to be computed during runtime
    size_t mini_batch_size_ = 0; // mini batch size to be computed during runtime
    size_t num_iterations_  = 0; // number of iterations to be computed during runtime
    size_t buffer_capacity_ = 2; // Size of the trajectory buffer
    int64_t max_return_     = INT64_MIN;

    std::tuple<size_t, size_t, size_t> obs_size_;

    CNNPolicyNetwork policy_net;
    CNNValueNetwork value_net;

    std::unique_ptr<torch::optim::Adam> policy_optimizer;
    std::unique_ptr<torch::optim::Adam> value_optimizer;

    std::unique_ptr<ReplayBuffer> buffer;

    torch::Tensor batch_indices_;

  public:
    PPO(std::tuple<size_t, size_t, size_t> obs_size, int action_size, PPOParams params) : params_(params), obs_size_(obs_size)
    {

        policy_net = CNNPolicyNetwork(obs_size, action_size);
        value_net  = CNNValueNetwork(obs_size);

        policy_optimizer = std::make_unique<torch::optim::Adam>(policy_net->parameters(), torch::optim::AdamOptions(params_.learning_rate));

        value_optimizer = std::make_unique<torch::optim::Adam>(value_net->parameters(), torch::optim::AdamOptions(params_.learning_rate));

        batch_size_      = params_.num_envs * params_.num_steps;
        mini_batch_size_ = batch_size_ / params_.num_minibatches;

        buffer = std::make_unique<ReplayBuffer>(buffer_capacity_);

        batch_indices_ = torch::arange((int64_t)batch_size_);

        load_networks("artifacts/");
    };

    void store_trajectory(const TrajectoryBuffer& trajectory) { buffer->add_trajectory(trajectory); }

    void clear_buffer() { buffer->clear(); }

    std::tuple<double, double> calculate_clip_fraction(torch::Tensor& log_ratio, torch::Tensor& ratio)
    {
        torch::NoGradGuard no_grad;
        // auto old_approx_kl = (-log_ratio).mean();

        auto approx_kl = ((ratio - 1) - log_ratio).mean();

        auto clipped_mask = (ratio - 1.0).abs() > params_.clip_epsilon;

        auto clipped_float = clipped_mask.to(torch::kFloat);

        auto clip_frac_tensor = clipped_float.mean();

        double clip_frac = clip_frac_tensor.item<double>();

        return {clip_frac, approx_kl.item<double>()};
    }

    void save_networks(const std::string& folder_path)
    {
        std::filesystem::create_directories(folder_path);

        {
            torch::serialize::OutputArchive policy_archive;
            policy_net->save(policy_archive);
            policy_archive.save_to(folder_path + "/policy.pt");
        }

        {
            torch::serialize::OutputArchive value_archive;
            value_net->save(value_archive);
            value_archive.save_to(folder_path + "/value.pt");
        }

        std::cout << "\033[31m>>>>>>>>>>>Saved networks to: " << folder_path << std::endl;
    }

    void load_networks(const std::string& folder_path)
    {
        if (!std::filesystem::exists(folder_path + "/policy.pt") || !std::filesystem::exists(folder_path + "/value.pt")) {
            std::cout << "\033[31m>>>>>>>>>>>No saved networks found at " << folder_path << std::endl;
            return;
        }

        torch::serialize::InputArchive policy_archive;
        policy_archive.load_from(folder_path + "/policy.pt");
        policy_net->load(policy_archive);

        torch::serialize::InputArchive value_archive;
        value_archive.load_from(folder_path + "/value.pt");
        value_net->load(value_archive);

        std::cout << "\033[31m>>>>>>>>>>>Loaded policy & value networks from: " << folder_path << std::endl;
    }

    void print_tensor(torch::Tensor& t)
    {
        for (int i = 0; i < t.sizes()[0]; ++i) {
            std::cout << t[i] << " ";
        }
        std::cout << std::endl;
    }

    void optimize(const TrajectoryBuffer& tb, torch::Tensor& advantages, torch::Tensor& returns)
    {
        double mean_pg_loss  = 0.0;
        double mean_v_loss   = 0.0;
        double mean_entropy  = 0.0;
        double mean_kl       = 0.0;
        double mean_clipfrac = 0.0;

        size_t update_count = 0;

        size_t channel, width, height;
        std::tie(channel, width, height) = obs_size_;
        // Perform PPO optimization step
        auto states      = tb.get_states();
        auto actions     = tb.get_actions();
        auto rewards     = tb.get_rewards();
        auto dones       = tb.get_dones();
        auto next_states = tb.get_next_states();

        auto log_probs = tb.get_log_probs();
        auto values    = tb.get_values();

        std::cout << "\033[31m>>>>>>>>>>>[ADV] mean=" << advantages.mean().item<double>() << " std=" << advantages.std().item<double>() << std::endl;

        // Normalize advantages
        auto norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8);

        // TODO probably reshape the states

        std::vector<double> clip_fractions;
        auto b_indices = batch_indices_.clone();
        for (size_t epoch = 0; epoch < params_.ppo_epochs; ++epoch) {

            // Generate a random permutation of indices
            auto perm = torch::randperm(b_indices.size(0));

            // Index the tensor with the random indices
            b_indices = b_indices.index({perm});

            for (size_t start = 0; start < batch_size_; start += mini_batch_size_) {
                auto end                = start + mini_batch_size_;
                auto mini_batch_indices = b_indices.slice(0, start, end);

                auto mb_states     = states.index({mini_batch_indices}).detach();
                auto mb_action     = actions.index({mini_batch_indices}).detach();
                auto mb_values     = values.index({mini_batch_indices}).detach();
                auto mb_returns    = returns.index({mini_batch_indices}).detach();
                auto mb_advantages = norm_advantages.index({mini_batch_indices}).detach();
                auto mb_log_probs  = log_probs.index({mini_batch_indices}).detach();

                auto [new_action, new_log_prob, entropy] = policy_net->get_actions(mb_states, mb_action);
                auto new_value                           = value_net->get_value(mb_states);

                auto log_ratio = new_log_prob - mb_log_probs;

                auto ratio = log_ratio.exp();

                auto [clip_frac, approx_kl] = calculate_clip_fraction(log_ratio, ratio);
                clip_fractions.push_back(clip_frac);

                // Policy gradient loss
                auto policy_gradient_loss         = -mb_advantages * ratio;
                auto policy_gradient_loss_clipped = -mb_advantages * torch::clamp(ratio, 1.0 - params_.clip_epsilon, 1.0 + params_.clip_epsilon);

                auto pg_loss = torch::max(policy_gradient_loss, policy_gradient_loss_clipped).mean();

                // std::cout << "PG LOSS : " << pg_loss << "\npolicy_gradient_loss : " << policy_gradient_loss
                //           << "\npolicy_gradient_loss_clipped : " << policy_gradient_loss_clipped << std::endl;
                // std::cout << "mb_advantages : " << mb_advantages << " ratio : " << ratio << std::endl;

                // Value loss
                new_value = new_value.view(-1);
                torch::Tensor v_loss;
                if (params_.clip_vloss) {
                    // clipped value prediction
                    auto v_clipped = mb_values + torch::clamp(new_value - mb_values, -params_.clip_epsilon, params_.clip_epsilon);

                    auto v_loss_unclipped = torch::pow(new_value - mb_returns, 2);
                    auto v_loss_clipped   = torch::pow(v_clipped - mb_returns, 2);

                    // take the max
                    auto v_loss_max = torch::max(v_loss_unclipped, v_loss_clipped);
                    v_loss          = 0.5 * v_loss_max.mean();
                } else {
                    auto v_loss_raw = torch::pow(new_value - mb_returns, 2);
                    v_loss          = 0.5 * v_loss_raw.mean();
                }

                auto entropy_loss = entropy.mean();

                auto loss_policy = pg_loss + entropy_loss * params_.ent_coef;
                auto loss_value  = v_loss * params_.vf_coef;
                auto total_loss  = loss_policy + loss_value;

                value_optimizer->zero_grad();
                policy_optimizer->zero_grad();

                total_loss.backward();

                torch::nn::utils::clip_grad_norm_(policy_net->parameters(), params_.max_grad_norm);
                torch::nn::utils::clip_grad_norm_(value_net->parameters(), params_.max_grad_norm);

                policy_optimizer->step();
                value_optimizer->step();

                mean_pg_loss += pg_loss.item<double>();
                mean_v_loss += v_loss.item<double>();
                mean_entropy += entropy_loss.item<double>();
                mean_kl += approx_kl;
                mean_clipfrac += clip_frac;
                update_count++;

                if (approx_kl > params_.target_kl) {
                    std::cout << "\033[38m>>>>>>>>>>>[EARLY STOP]"
                              << " kl=" << approx_kl << std::endl;
                    break;
                }
            }
            std::cout << "\033[37m>>>>>>>>>>>[PPO UPDATE]"
                      << " pg_loss=" << mean_pg_loss / update_count << " v_loss=" << mean_v_loss / update_count
                      << " entropy=" << mean_entropy / update_count << " kl=" << mean_kl / update_count
                      << " clip_frac=" << mean_clipfrac / update_count << std::endl;
        }

        // Publish some statistics
    }

    std::tuple<torch::Tensor, torch::Tensor> calculate_advantages_and_returns(const TrajectoryBuffer& tb)
    {
        torch::NoGradGuard no_grad;

        auto states      = tb.get_states().detach();
        auto actions     = tb.get_actions().detach();
        auto rewards     = tb.get_rewards().detach();
        auto dones       = tb.get_dones().detach();
        auto next_states = tb.get_next_states().detach();
        auto log_probs   = tb.get_log_probs().detach();
        auto values      = tb.get_values().detach();

        auto next_values = value_net->get_value(next_states).detach();
        auto advantages  = torch::zeros_like(next_values);

        torch::Tensor last_gae = torch::zeros_like(values[0]);
        torch::Tensor next_non_terminal;
        torch::Tensor curr_next_value;

        auto last_done       = dones[dones.size(0) - 1];
        auto last_next_value = next_values[next_values.size(0) - 1];

        for (int64_t t = params_.num_steps - 1; t >= 0; --t) {

            if (t == params_.num_steps - 1) {
                // Special scenario that I need to understand later
                next_non_terminal = 1.0 - last_done;
                curr_next_value   = last_next_value;
            } else {
                next_non_terminal = 1.0 - dones[t + 1];
                curr_next_value   = values[t + 1];
            }

            // r + gamma * v'[t] * done - v[t]
            // advantage[t] = delta + gamma * gae_lambda * done * last_gae
            auto delta = rewards[t] + params_.gamma * curr_next_value * next_non_terminal - values[t];
            // std::cout << "Delta at time " << t << " : " << delta << std::endl;
            advantages[t] = delta + params_.gamma * params_.gae_lambda * next_non_terminal * last_gae;

            last_gae = advantages[t];
        }

        auto returns = advantages + values;

        // Ensure no gradient escapes
        advantages = advantages.detach();
        std::cout << "Advantages: " << advantages << std::endl;
        returns = returns.detach();
        std::cout << "returns: " << returns << std::endl;

        auto value_error = (returns - values).pow(2).mean();

        std::cout << "\033[31m>>>>>>>>>>>[VALUE] mse=" << value_error.item<double>() << std::endl;

        return {advantages, returns};
    }

    void collect(Env& env)
    {
        torch::NoGradGuard no_grad;

        TrajectoryBuffer trajectory_buffer(params_.num_steps);

        auto state            = env.reset(); // reset returns initial state
        double episode_return = 0.0;
        size_t episode_length = 0;

        std::cout << "\033[32m>>>>>>>>>>> Collecting a trajectory from rollouts" << std::endl;

        for (size_t t = 0; t < params_.num_steps; ++t) {
            auto [action, log_prob, entropy] = policy_net->get_actions(state);
            auto [next_state, reward, done]  = env.step(action.squeeze(0));
            auto value                       = value_net->forward(state).flatten();

            // Add transition to buffer
            trajectory_buffer.add(state.squeeze(0),
                                  action,
                                  log_prob,
                                  torch::full({(int64_t)params_.num_envs}, reward, state.options()),
                                  next_state.squeeze(0),
                                  torch::full({(int64_t)params_.num_envs}, done ? 1.0 : 0.0, state.options()),
                                  value);

            episode_return += reward;
            episode_length++;

            state = next_state;

            // If done: break early — end of episode
            if (done) {
                break;
            }
        }

        buffer->add_trajectory(trajectory_buffer);

        if (episode_return > max_return_) {
            save_networks("artifacts");
        }

        std::cout << "\033[31m>>>>>>>>>>>[ROLL OUT] return=" << episode_return << " length=" << episode_length << std::endl;
    }

    void anneal_lr(size_t episode, size_t max_episodes)
    {
        auto frac   = 1.0 - ((double)episode - 1.0) / max_episodes;
        auto new_lr = frac * params_.learning_rate;

        std::cout << "\033[32m>>>>>>>>>>> Annealing the learning rate. Starting Value (LR) : " << params_.learning_rate
                  << " - New Value (LR) : " << new_lr << std::endl;

        // Update optimizers with new learning rate
        for (auto& opt : {policy_optimizer.get(), value_optimizer.get()}) {
            for (auto& group : opt->param_groups()) {
                static_cast<torch::optim::AdamOptions&>(group.options()).lr(new_lr);
            }
        }
    }

    torch::Tensor act(torch::Tensor& state)
    {
        // Get action from policy network
        auto [action, log_prob, entropy] = policy_net->get_actions(state.unsqueeze(0));
        return action.squeeze(0);
    }

    void train(Env& env, size_t tb_count)
    {
        std::cout << "\033[32m>>>>>>>>>>> Training has started!" << std::endl;
        for (size_t episode = 0; episode < params_.total_time_steps; ++episode) {
            std::cout << "\033[33m>>>>>>>>>>> Episode : " << episode + 1 << std::endl;
            anneal_lr(episode, params_.total_time_steps);
            collect(env);

            if (tb_count > 1) {
                std::cout << "\033[34m>>>>>>>>>>> Sampling trajectories (N-TB) : " << tb_count << std::endl;
                auto trajectories = buffer->sample_tbs(tb_count);
                size_t tb_id      = 0;
                for (const auto& tb_ref : trajectories) {
                    std::cout << "\033[35m>>>>>>>>>>> Optimization Trajectory " << tb_id << std::endl;

                    const auto& tb             = tb_ref.get();
                    auto [advantages, returns] = calculate_advantages_and_returns(tb);
                    optimize(tb, advantages, returns);

                    tb_id++;
                }
            } else {
                std::cout << "\033[34m>>>>>>>>>>> Get Last Trajectory" << std::endl;
                TrajectoryBuffer tb = buffer->get_last(); // For now only last

                std::cout << "\033[34m>>>>>>>>>>> Calculate Advantages and Returns!" << std::endl;
                auto [advantages, returns] = calculate_advantages_and_returns(tb);

                std::cout << "\033[34m>>>>>>>>>>> Optimization Step!" << std::endl;
                optimize(tb, advantages, returns);
            }
            std::cout << "\033[35m------------------------------------ END OF AN EPISODE ------------------------------------ " << std::endl;
        }
    }

    ~PPO() = default;
};
