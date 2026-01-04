#pragma once

#include <cmath>

#include <torch/torch.h>

#include "distributions.hpp"
#include "networks.hpp"

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

        // Calculate the features dynamically
        auto dummy     = torch::zeros({1, (int64_t)std::get<0>(obs_size), (int64_t)std::get<1>(obs_size), (int64_t)std::get<2>(obs_size)});
        auto n_flatten = seq_cnn->forward(dummy).numel();

        seq_linear = register_module("seq_linear",
                                     torch::nn::Sequential(torch::nn::Flatten(),
                                                           layer_init(torch::nn::Linear(n_flatten, 512)),
                                                           torch::nn::Tanh(),
                                                           layer_init(torch::nn::Linear(512, action_dim))));

        // nn.Parameter equivalent
        actor_log_std = register_parameter("actor_log_std", torch::full({static_cast<long long>(action_dim)}, -3.0));
    }

    torch::Tensor forward(const torch::Tensor& x)
    {
        auto cnn_features = seq_cnn->forward(x);
        return seq_linear->forward(cnn_features);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_actions(const torch::Tensor& x,
                                                                        c10::optional<torch::Tensor> action_opt = c10::nullopt)
    {
        auto action_means = forward(x);
        CategoricalDist dist(action_means);

        torch::Tensor action_out;
        if (action_opt.has_value()) {
            action_out = action_opt.value();
        } else {
            action_out = dist.sample();
        }

        auto log_prob = dist.log_prob(action_out);

        auto entropy = dist.entropy();

        return {action_out, log_prob, entropy};
    }

    torch::nn::Sequential seq_cnn    = nullptr;
    torch::nn::Sequential seq_linear = nullptr;
    torch::Tensor actor_log_std;
};

TORCH_MODULE(CNNPolicyNetwork);
