#pragma once

#include "agent/a2c.hpp"
#include "agent/policy.hpp"
#include "agent/value.hpp"
#include "buffer.hpp"
#include "ds.hpp"
#include "envs.hpp"

#include <filesystem>
#include <iostream>
#include <memory>
#include <torch/torch.h>
#include <tuple>

typedef struct PPO_RNN_PARAMS {
    size_t total_time_steps     = 1000000; // total timesteps of the experiments
    double learning_rate_policy = 1e-4;    // the learning rate of the optimizer
    double learning_rate_critic = 1e-5;    // the learning rate of the optimizer
    double min_lr               = 1e-5;    // minumum value of learning rate
    size_t num_envs             = 1;       // the number of parallel game environments. 1 for now!
    size_t num_steps            = 2048;    // the number of steps to run in each environment per policy rollout
    bool anneal_lr              = false;   // Toggle learning rate annealing for policy and value networks
    double gamma                = 0.99;    // the discount factor gamma
    double gae_lambda           = 0.95;    // the lambda for the general advantage estimation
    size_t num_minibatches      = 32;      // the number of mini-batches
    size_t ppo_epochs           = 10;      // the K epochs to update the policy
    bool norm_adv               = true;    // Toggles advantages normalization
    double clip_epsilon         = 0.2;     // the surrogate clipping coefficient
    bool clip_vloss             = true;    // Toggles whether or not to use a clipped loss for the value function, as per the paper.
    double ent_coef             = 0.0;     // coefficient of the entropy
    double vf_coef              = 0.5;     // coefficient of the value function
    double max_grad_norm        = 0.5;     // the maximum norm for the gradient clipping
    double target_kl            = 0.02;    // the target KL divergence threshold
    double reward_sharper       = 0.01;    // Reward scaling parameter
    size_t grad_acc_steps       = 2;       // Gradient accumulation step counts
} PPO_RNN_PARAMS;

class PPO_RNN {
  private:
    PPO_RNN_PARAMS params_;

    size_t batch_size_                = 0; // batch_size to be computed during runtime
    size_t mini_batch_size_           = 0; // mini batch size to be computed during runtime
    size_t num_iterations_            = 0; // number of iterations to be computed during runtime
    size_t buffer_capacity_           = 2; // Size of the trajectory buffer
    double max_return_rollout_return_ = INT32_MIN;
    double max_return_                = INT32_MIN;
    size_t current_episode_           = 0;
    size_t log_rate_                  = 1;  // Show metrics every N step
    size_t track_last_n               = 10; // Track the progress of past N steps
    size_t best_checkpoint_episode    = 0;  // Best episode index to be able to save the model
    size_t save_top_n                 = 5;  // Only store top N epoch
    bool separate_nets                = false;

    std::unique_ptr<CircularBuffer<double>> last_n_returns;    // We want to store the last N returns to calculate avg return
    std::unique_ptr<CircularBuffer<size_t>> saved_checkpoints; // To be able to efficiently remove the old checkpoint we need this

    std::tuple<size_t, size_t, size_t> obs_size_;

    RNNActorCriticNetwork actor_critic_net;

    std::unique_ptr<torch::optim::Adam> a2c_optimizer;

    std::unique_ptr<ReplayBuffer> buffer;

    std::tuple<torch::Tensor, torch::Tensor> next_lstm_state_; // Global lstm state to track the historical information
    torch::Tensor next_done_;                                  // Global done tensor to support lstm state calculations mainly for the masking

  public:
    PPO_RNN(std::tuple<size_t, size_t, size_t> obs_size, int action_size, PPO_RNN_PARAMS params, bool separate_nets)
        : params_(params), obs_size_(obs_size), separate_nets(separate_nets)
    {
        actor_critic_net = RNNActorCriticNetwork(obs_size, action_size);
        a2c_optimizer = std::make_unique<torch::optim::Adam>(actor_critic_net->parameters(), torch::optim::AdamOptions(params_.learning_rate_policy));

        batch_size_      = params_.num_envs * params_.num_steps;
        mini_batch_size_ = batch_size_ / params_.num_minibatches;

        auto [lstm_layers, lstm_hidden_size] = actor_critic_net->get_lstm_info();
        next_lstm_state_ = std::make_tuple(torch::zeros({lstm_layers, 1, lstm_hidden_size}), torch::zeros({lstm_layers, 1, lstm_hidden_size}));
        next_done_       = torch::zeros({1});

        buffer = std::make_unique<ReplayBuffer>(buffer_capacity_);

        load_networks("artifacts_rnn");

        last_n_returns    = std::make_unique<CircularBuffer<double>>(track_last_n);
        saved_checkpoints = std::make_unique<CircularBuffer<size_t>>(save_top_n);
    };

    void store_trajectory(const TrajectoryBuffer& trajectory) { buffer->add_trajectory(trajectory); }

    void clear_buffer() { buffer->clear(); }

    std::tuple<double, double> calculate_clip_fraction(torch::Tensor& log_ratio, torch::Tensor& ratio)
    {
        torch::NoGradGuard no_grad;

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

        torch::serialize::OutputArchive a2c_archive;
        actor_critic_net->save(a2c_archive);
        a2c_archive.save_to(folder_path + "/a2c_" + std::to_string(best_checkpoint_episode) + ".pt");

        if (current_episode_ % log_rate_ == 0) {
            std::cout << "\033[31m>>>>>>>>>>>Saved networks to: " << folder_path << std::endl;
        }
    }

    void load_networks(const std::string& folder_path)
    {
        if (!std::filesystem::exists(folder_path)) {
            std::cout << "\033[31m>>>>>>>>>>>No saved networks found at " << folder_path << std::endl;
            return;
        }

        if (!std::filesystem::is_directory(folder_path)) {
            std::cout << "\033[31m>>>>>>>>>>>Directory check failed for " << folder_path << std::endl;
            return;
        }
        int64_t last_epoch = -1;
        for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
            // Only include regular files
            if (entry.is_regular_file()) {
                std::string file_path   = entry.path().filename().string();
                std::string episode_str = split(file_path, "_")[1];
                episode_str             = split(episode_str, ".")[0];
                int64_t episode_id      = std::stoi(episode_str);
                last_epoch              = std::max(last_epoch, episode_id);
            }
        }

        torch::serialize::InputArchive a2c_archive;
        a2c_archive.load_from(folder_path + "/a2c_" + std::to_string(last_epoch) + ".pt");
        actor_critic_net->load(a2c_archive);

        std::cout << "\033[31m>>>>>>>>>>>Loaded policy & value networks from: " << folder_path << "\nLoaded epoch index :" << last_epoch << std::endl;
    }

    void print_tensor(torch::Tensor& t)
    {
        for (int i = 0; i < t.sizes()[0]; ++i) {
            std::cout << t[i] << " ";
        }
        std::cout << std::endl;
    }

    void remove_old_checkpoints(const std::string& folder_path)
    {
        if (!saved_checkpoints->is_full())
            return;

        size_t old_ep = saved_checkpoints->pop_first();

        std::filesystem::path model_path = folder_path + "/a2c_" + std::to_string(old_ep) + ".pt";
        try {
            std::filesystem::remove(model_path);
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error deleting file: " << e.what() << "\n";
        }
    }

    void reset_rnn_states()
    {
        auto [lstm_layers, lstm_hidden_size] = actor_critic_net->get_lstm_info();
        next_lstm_state_ = std::make_tuple(torch::zeros({lstm_layers, 1, lstm_hidden_size}), torch::zeros({lstm_layers, 1, lstm_hidden_size}));
        next_done_       = torch::zeros({1});
    }

    static torch::Tensor compute_episode_starts_from_dones(const torch::Tensor& dones, // shape [T * N]
                                                           int64_t T,
                                                           int64_t N)
    {
        // dones[t] = terminal after executing action at t
        // episode_start[t] should be terminal from previous step: dones_out[t-1]
        // so we can rebuild lstm states from scratch
        auto d      = dones.view({T, N});
        auto zeros  = torch::zeros({1, N}, d.options());
        auto starts = torch::cat({zeros, d.slice(/*dim=*/0, /*start=*/0, /*end=*/T - 1)}, /*dim=*/0);
        return starts.flatten(); // [T * N]
    }

    void optimize(const TrajectoryBuffer& tb, torch::Tensor& advantages, torch::Tensor& returns)
    {
        double mean_pg_loss  = 0.0;
        double mean_v_loss   = 0.0;
        double mean_entropy  = 0.0;
        double mean_kl       = 0.0;
        double mean_clipfrac = 0.0;

        size_t update_count = 0;

        auto initial_h_state    = std::get<0>(next_lstm_state_).clone();
        auto initial_c_state    = std::get<1>(next_lstm_state_).clone();
        auto initial_lstm_state = std::make_tuple(initial_h_state, initial_c_state);

        size_t channel, width, height;
        std::tie(channel, width, height) = obs_size_;

        const int64_t T = (int64_t)params_.num_steps;
        const int64_t N = (int64_t)params_.num_envs;

        // Perform PPO optimization step
        auto states      = tb.get_states();                // [B=T*N, C, H, W]
        auto actions     = tb.get_actions();               // [B=T*N, num_actions]
        auto rewards     = tb.get_rewards().squeeze(-1);   // [B=T*N]
        auto dones       = tb.get_dones().squeeze(-1);     // [B=T*N]
        auto next_states = tb.get_next_states();           // [B=T*N, C, H, W]
        auto log_probs   = tb.get_log_probs().squeeze(-1); // [B=T*N]
        auto values      = tb.get_values().squeeze(-1);    // [B=T*N]

        TORCH_CHECK(states.size(0) == T * N, "states first dim must be T*N");
        TORCH_CHECK(dones.size(0) == T * N, "dones first dim must be T*N");

        if (current_episode_ % log_rate_ == 0) {
            std::cout << "\033[31m>>>>>>>>>>>[ADV] mean=" << advantages.mean().item<double>() << " std=" << advantages.std().item<double>()
                      << std::endl;
        }

        auto episode_starts = compute_episode_starts_from_dones(dones, T, N); // [T * N]

        // Normalize advantages
        torch::Tensor used_advantages = advantages;
        if (params_.norm_adv) {
            used_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8);
        }

        // Ensure correct shape assumptions
        TORCH_CHECK(states.dim() == 4, "states must be [T,C,H,W] when num_envs=1");
        TORCH_CHECK(states.size(0) == T, "states length must equal T");

        // Actions: [T] (Categorical) or [T,K]
        TORCH_CHECK(actions.size(0) == T, "actions length must equal T");

        std::vector<double> clip_fractions;
        uint64_t steps    = 0;
        bool stop_updates = false;

        // This is pretty much max environment length since we have a single environment
        const int64_t chunk_len              = std::max<int64_t>(1, (int64_t)mini_batch_size_); // e.g. 64
        auto [lstm_layers, lstm_hidden_size] = actor_critic_net->get_lstm_info();

        for (size_t epoch = 0; epoch < params_.ppo_epochs; ++epoch) {
            if (current_episode_ % log_rate_ == 0) {
                std::cout << "\033[37m>>>>>>>>>>>[PPO EPOCH] " << epoch + 1 << "/" << params_.ppo_epochs << std::endl;
            }
            // Reconstruct from zeros each epoch (rollout starts from reset_rnn_states()) :contentReference[oaicite:4]{index=4}
            auto h_state = torch::zeros({lstm_layers, 1, lstm_hidden_size}, states.options());
            auto c_state = torch::zeros({lstm_layers, 1, lstm_hidden_size}, states.options());

            for (int64_t start = 0; start < T; start += chunk_len) {
                // We slice it max_env_length by max_env_length
                int64_t end = std::min<int64_t>(start + chunk_len, T);

                auto mb_states     = states.slice(0, start, end).detach();         // [L,C,H,W]
                auto mb_log_probs  = log_probs.slice(0, start, end).detach();      // [L]
                auto mb_values     = values.slice(0, start, end).detach();         // [L]
                auto mb_returns    = returns.slice(0, start, end).detach();        // [L]
                auto mb_advantages = advantages.slice(0, start, end).detach();     // [L]
                auto mb_ep_start   = episode_starts.slice(0, start, end).detach(); // [L]

                torch::Tensor mb_actions = actions.slice(0, start, end).detach();

                // For categorical, actions must be int64
                if (mb_actions.scalar_type() != torch::kLong) {
                    mb_actions = mb_actions.to(torch::kLong);
                }

                auto res = actor_critic_net->get_actions(mb_states, std::make_tuple(h_state, c_state), mb_ep_start, mb_actions);

                auto new_action   = std::get<0>(res).squeeze(-1); // [L]
                auto new_log_prob = std::get<1>(res).squeeze(-1); // [L]
                auto entropy      = std::get<2>(res).squeeze(-1); // [L]
                auto new_value    = std::get<3>(res).squeeze(-1); // [L]
                auto new_state    = std::get<4>(res);

                h_state = std::get<0>(new_state);
                c_state = std::get<1>(new_state);

                auto log_ratio = new_log_prob - mb_log_probs; // [MB]

                auto ratio = log_ratio.exp(); // [MB]

                if (epoch == 0 && start == 0) {
                    auto lr_mean   = (new_log_prob - mb_log_probs).mean().item<double>();
                    auto lr_std    = (new_log_prob - mb_log_probs).std().item<double>();
                    auto ratio_min = ratio.min().item<double>();
                    auto ratio_max = ratio.max().item<double>();

                    std::cout << "\033[31m>>>>>>>>>>>[DEBUG] log_ratio mean=" << lr_mean << " std=" << lr_std << " ratio min=" << ratio_min
                              << " max=" << ratio_max << std::endl;

                    std::cout << "\033[31m>>>>>>>>>>>[DEBUG] mb_action dtype=" << mb_actions.dtype() << " shape=" << mb_actions.sizes()
                              << " mb_log_probs shape=" << mb_log_probs.sizes() << " new_log_prob shape=" << new_log_prob.sizes() << std::endl;
                    start = false;
                }

                auto [clip_frac, approx_kl] = calculate_clip_fraction(log_ratio, ratio);
                clip_fractions.push_back(clip_frac);

                // Policy gradient loss
                auto policy_gradient_loss = mb_advantages * ratio; // [MB]

                auto policy_gradient_loss_clipped =
                    mb_advantages * torch::clamp(ratio, 1.0 - params_.clip_epsilon, 1.0 + params_.clip_epsilon); // [MB]

                auto pg_loss = -torch::min(policy_gradient_loss, policy_gradient_loss_clipped).mean(); // [] single value

                // Value loss
                new_value = new_value.view(-1); // [MB]

                torch::Tensor v_loss;
                if (params_.clip_vloss) {
                    // clipped value prediction
                    auto v_clipped = mb_values + torch::clamp(new_value - mb_values, -params_.clip_epsilon, params_.clip_epsilon); // [MB]

                    auto v_loss_unclipped = torch::pow(new_value - mb_returns, 2); // [MB]

                    auto v_loss_clipped = torch::pow(v_clipped - mb_returns, 2); // [MB]

                    // take the max
                    auto v_loss_max = torch::max(v_loss_unclipped, v_loss_clipped); // [MB]

                    v_loss = 0.5 * v_loss_max.mean(); // []

                } else {

                    auto v_loss_raw = torch::pow(new_value - mb_returns, 2);
                    v_loss          = 0.5 * v_loss_raw.mean();
                }

                auto entropy_loss = -entropy.mean(); // []

                auto loss_policy = pg_loss + entropy_loss * params_.ent_coef;
                auto loss_value  = v_loss * params_.vf_coef;
                auto total_loss  = loss_policy + loss_value;

                // Scale the total loss according to the gradient accumulation steps
                if (params_.grad_acc_steps > 1) {
                    total_loss = total_loss / (double)params_.grad_acc_steps;
                }

                total_loss.backward();

                steps++;

                if (steps % params_.grad_acc_steps == 0) {
                    torch::nn::utils::clip_grad_norm_(actor_critic_net->parameters(), params_.max_grad_norm);
                    a2c_optimizer->step();
                    a2c_optimizer->zero_grad();
                }

                mean_pg_loss += pg_loss.item<double>();
                mean_v_loss += v_loss.item<double>();
                mean_entropy += entropy_loss.item<double>();
                mean_kl += approx_kl;
                mean_clipfrac += clip_frac;
                update_count++;

                if (params_.target_kl > 0.0 && approx_kl > 1.5 * params_.target_kl) {
                    if (current_episode_ % log_rate_ == 0) {
                        std::cout << "\033[38m>>>>>>>>>>>[EARLY STOP] kl=" << approx_kl << std::endl;
                    }
                    stop_updates = true;
                    break; // break minibatch loop
                }
            }

            // If there is any remaining updates that hasn't been done, flush it here!
            if (steps % params_.grad_acc_steps != 0) {
                torch::nn::utils::clip_grad_norm_(actor_critic_net->parameters(), params_.max_grad_norm);
                a2c_optimizer->step();
                a2c_optimizer->zero_grad();
            }

            if (stop_updates) {
                break;
            }

            if (current_episode_ % log_rate_ == 0) {
                std::cout << "\033[37m>>>>>>>>>>>[PPO UPDATE]"
                          << " pg_loss=" << mean_pg_loss / update_count << " v_loss=" << mean_v_loss / update_count
                          << " entropy=" << mean_entropy / update_count << " kl=" << mean_kl / update_count
                          << " clip_frac=" << mean_clipfrac / update_count << std::endl;
            }
        }
    }

    std::tuple<torch::Tensor, torch::Tensor> calculate_advantages_and_returns(const TrajectoryBuffer& tb)
    {
        torch::NoGradGuard no_grad;

        auto rewards         = tb.get_rewards().squeeze(-1);    // [T]
        auto dones           = tb.get_dones().squeeze(-1);      // [T]
        auto values          = tb.get_values().squeeze(-1);     // [T]
        auto terminates      = tb.get_terminates().squeeze(-1); // [T] true terminal
        auto truncates       = tb.get_truncates().squeeze(-1);  // [T] time limit
        auto next_states     = tb.get_next_states();
        auto terminal_values = tb.get_terminal_values().squeeze(-1); // [T]  (add this)

        auto idx = next_states.size(0) - 1;

        auto episode_start = torch::zeros({1}, rewards.options());
        auto next_value =
            actor_critic_net->get_value(next_states.index({idx}).unsqueeze(0), next_lstm_state_, episode_start).detach().squeeze(-1); // [1]

        int64_t T = rewards.size(0);

        torch::Tensor advantages = torch::zeros_like(values).unsqueeze(-1); // [T, 1]
        torch::Tensor last_gae   = torch::zeros_like(values[0]);            // [], just a value

        for (int64_t t = T - 1; t >= 0; --t) {
            auto gae_mask       = 1.0 - dones[t];      // [T]
            auto bootstrap_mask = 1.0 - terminates[t]; // [T]

            // next value: either bootstrap or the next in values
            auto next_val = (t == T - 1 ? next_value : values[t + 1]); // [1]

            // if the episode ended here:
            // - terminate : stop here no bootstrapping
            // - truncated: bootstrap from terminal_values[t]
            if (dones[t].item<double>() > 0.5) {
                next_val = torch::where(truncates[t] > 0.5, terminal_values[t], torch::zeros_like(next_val));
            }

            // delta = r + γ * V(next) * mask - V(current)
            auto delta = rewards[t] + params_.gamma * next_val * bootstrap_mask - values[t]; // [1]

            // GAE recurrence
            last_gae = delta + params_.gamma * params_.gae_lambda * gae_mask * last_gae; // [1]

            advantages[t] = last_gae;
        }

        advantages.squeeze_(-1);            // [T]
        auto returns = advantages + values; // [T]

        auto value_error = (returns - values).pow(2).mean();

        if (current_episode_ % log_rate_ == 0) {
            std::cout << "\033[31m>>>>>>>>>>>[VALUE] mse=" << value_error.item<double>() << std::endl;
        }

        return {advantages.detach(), returns.detach()};
    }

    void collect(Env& env, size_t episode_index)
    {
        torch::NoGradGuard no_grad;

        TrajectoryBuffer trajectory_buffer(params_.num_steps);

        auto state                       = env.reset(); // reset returns initial state
        double episode_return            = 0.0;
        size_t completed_episodes        = 0;
        double completed_episode_returns = 0;

        if (current_episode_ % log_rate_ == 0) {
            std::cout << "\033[32m>>>>>>>>>>> Collecting a trajectory from rollouts" << std::endl;
        }

        for (size_t t = 0; t < params_.num_steps; ++t) {
            torch::Tensor action, log_prob, entropy, value;

            auto res         = actor_critic_net->get_actions(state, next_lstm_state_, next_done_);
            action           = std::get<0>(res);
            log_prob         = std::get<1>(res);
            entropy          = std::get<2>(res);
            value            = std::get<3>(res).flatten();
            next_lstm_state_ = std::get<4>(res);

            auto [next_state, reward, done, terminated, truncated] = env.step(action.squeeze(0));

            auto tb_state      = state.squeeze(0);
            auto tb_next_state = next_state.squeeze(0);

            auto scaled_reward = reward * params_.reward_sharper; // reward scaling

            auto reward_tensor     = torch::full({(int64_t)params_.num_envs}, scaled_reward, state.options());
            next_done_             = torch::full({(int64_t)params_.num_envs}, done ? 1.0 : 0.0, state.options());
            auto terminated_tensor = torch::full({(int64_t)params_.num_envs}, terminated ? 1.0 : 0.0, state.options());
            auto truncated_tensor  = torch::full({(int64_t)params_.num_envs}, truncated ? 1.0 : 0.0, state.options());

            episode_return += reward;

            torch::Tensor terminal_value = torch::zeros({1}, state.options());

            // If truncated boostrap the terminal value, this will be used while we are calculating the advantages
            if (truncated && !terminated) {
                auto episode_start0 = torch::zeros({1}, state.options());
                terminal_value =
                    actor_critic_net->get_value(tb_next_state.unsqueeze(0), next_lstm_state_, episode_start0).detach().squeeze(-1); // [1]
            }

            // Add transition to buffer
            trajectory_buffer.add(tb_state,
                                  action.squeeze(0),
                                  log_prob.squeeze(0),
                                  reward_tensor,
                                  tb_next_state,
                                  next_done_,
                                  terminated_tensor,
                                  truncated_tensor,
                                  value,
                                  terminal_value);

            // If done: break early — end of episode
            if (done) {
                completed_episodes++;
                completed_episode_returns += env.last_total_reward();
                state = env.reset();
            } else {
                state = next_state;
            }
        }

        buffer->add_trajectory(trajectory_buffer);

        last_n_returns->push(episode_return);

        auto sum_of_last_n         = last_n_returns->sum();
        max_return_rollout_return_ = std::max(max_return_rollout_return_, sum_of_last_n);

        // Save networks
        if (episode_return > max_return_) {
            best_checkpoint_episode = episode_index;
            max_return_             = episode_return;
            save_networks("artifacts_rnn");

            saved_checkpoints->push(episode_index);
            remove_old_checkpoints("artifacts_rnn");
        }

        if (current_episode_ % log_rate_ == 0) {
            double avg_episode = (completed_episodes > 0) ? (completed_episode_returns / (double)completed_episodes) : 0.0;

            std::cout << "\033[31m>>>>>>>>>>>[ROLL OUT] return=" << episode_return << " sum_of_last_" << track_last_n << "=" << sum_of_last_n
                      << "\nAVG Reward Per Episode : " << avg_episode << "\nBest Period : " << max_return_rollout_return_
                      << " Best Run Result : " << max_return_ << std::endl;
        }
    }

    void anneal_lr(size_t episode, size_t max_episodes)
    {
        if (!params_.anneal_lr)
            return;

        auto frac   = (double)episode / std::max<size_t>(1, max_episodes - 1);
        auto new_lr = frac * params_.learning_rate_policy;
        new_lr      = std::max(params_.min_lr, frac * params_.learning_rate_policy);

        // Update optimizers with new learning rate
        if (a2c_optimizer) {
            for (auto& group : a2c_optimizer->param_groups()) {
                static_cast<torch::optim::AdamOptions&>(group.options()).lr(new_lr);
            }
        }

        std::cout << "\033[32m>>>>>>>>>>> Annealing the learning rate. Starting Value (LR) : " << params_.learning_rate_policy
                  << " - New Value (LR) : " << new_lr << std::endl;
    }

    torch::Tensor act(torch::Tensor& state)
    {
        auto res         = actor_critic_net->get_actions(state, next_lstm_state_, next_done_);
        next_lstm_state_ = std::get<4>(res);
        return std::get<0>(res);
    }

    void train(Env& env, size_t tb_count)
    {
        std::cout << "\033[32m>>>>>>>>>>> Training has started!" << std::endl;
        size_t num_updates = (params_.total_time_steps + params_.num_steps - 1) / params_.num_steps;
        for (size_t update_episode = 0; update_episode < num_updates; ++update_episode) {
            reset_rnn_states();

            if (current_episode_ % log_rate_ == 0) {
                std::cout << "\033[33m>>>>>>>>>>> Episode : " << update_episode + 1 << std::endl;
                anneal_lr(update_episode, num_updates);
            }

            collect(env, update_episode);

            if (tb_count > 1) {
                auto trajectories = buffer->sample_tbs(tb_count);
                size_t tb_id      = 0;
                for (const auto& tb_ref : trajectories) {
                    const auto& tb             = tb_ref.get();
                    auto [advantages, returns] = calculate_advantages_and_returns(tb);
                    optimize(tb, advantages, returns);

                    tb_id++;
                }
            } else {
                TrajectoryBuffer tb = buffer->get_last(); // For now only last

                auto [advantages, returns] = calculate_advantages_and_returns(tb);

                optimize(tb, advantages, returns);
            }

            current_episode_++;

            if (current_episode_ % log_rate_ == 0) {
                std::cout << "\033[35m------------------------------------ END OF AN EPISODE ------------------------------------ " << std::endl;
            }
        }
        std::cout << "Max Sum of Last " << track_last_n << " : " << max_return_ << std::endl;
    }

    void play(Env& env, size_t max_steps)
    {
        torch::NoGradGuard no_grad;

        auto state = env.reset(); // reset returns initial state

        std::cout << "\033[32m>>>>>>>>>>> Playing the trained agent!" << std::endl;

        reset_rnn_states();

        for (size_t t = 0; t < max_steps; ++t) {
            auto action                                            = act(state);
            auto [next_state, reward, done, terminated, truncated] = env.step(action.squeeze(0));

            state = next_state;

            // If done: break early — end of episode
            if (done) {
                break;
            }
        }
    }

    ~PPO_RNN() = default;
};
