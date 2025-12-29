#pragma once

#include <cmath>
#include <networks.hpp>
#include <torch/torch.h>

struct CNNPolicyNetworkImpl : torch::nn::Module {

    CNNPolicyNetworkImpl() = default;

    CNNPolicyNetworkImpl(std::tuple<size_t, size_t, size_t> obs_size, size_t action_dim)
    {
        seq_cnn = register_module(
            "seq_cnn",
            torch::nn::Sequential(layer_init(torch::nn::Conv2d(torch::nn::Conv2dOptions(std::get<0>(obs_size), 32, 8).stride(4).bias(false))),
                                  torch::nn::ReLU(),
                                  layer_init(torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 4).stride(2).bias(false))),
                                  torch::nn::ReLU(),
                                  layer_init(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 2).stride(2).bias(false))),
                                  torch::nn::ReLU()));

        seq_linear = register_module("seq_linear",
                                     torch::nn::Sequential(torch::nn::Flatten(),
                                                           layer_init(torch::nn::Linear(128 * 6 * 6, 512)),
                                                           torch::nn::Tanh(),
                                                           layer_init(torch::nn::Linear(512, action_dim))));

        // nn.Parameter equivalent
        actor_log_std = register_parameter("actor_log_std", torch::zeros({static_cast<long long>(action_dim)}));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        auto cnn_features = seq_cnn->forward(x);
        return seq_linear->forward(cnn_features);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_actions(torch::Tensor x, c10::optional<torch::Tensor> action = c10::nullopt)
    {
        auto action_mean = forward(x);

        // Clamp for numerical stability (same as most PPO impls)
        auto action_logstd = torch::clamp(actor_log_std.expand_as(action_mean), -20.0, 2.0);

        auto action_std = torch::exp(action_logstd);

        // Sample action
        torch::Tensor action_sample;
        if (action.has_value()) {
            action_sample = action.value();
        } else {
            action_sample = action_mean + action_std * torch::randn_like(action_mean);
        }

        // Log probability (Normal)
        auto var      = action_std.pow(2);
        auto log_prob = -((action_sample - action_mean).pow(2)) / (2 * var) - action_logstd - 0.5 * std::log(2.0 * M_PI);

        log_prob = log_prob.sum(1);

        // Entropy
        auto entropy = (0.5 + 0.5 * std::log(2.0 * M_PI) + action_logstd).sum(1);

        return {action_sample, log_prob, entropy};
    }

    torch::nn::Sequential seq_cnn    = nullptr;
    torch::nn::Sequential seq_linear = nullptr;
    torch::Tensor actor_log_std;
};

TORCH_MODULE(CNNPolicyNetwork);
