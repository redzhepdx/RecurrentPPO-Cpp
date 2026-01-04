#pragma once

#include <torch/torch.h>

#include "networks.hpp"

struct CNNValueNetworkImpl : torch::nn::Module {
    CNNValueNetworkImpl() = default;
    CNNValueNetworkImpl(std::tuple<size_t, size_t, size_t> obs_size)
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

        seq_linear = register_module(
            "seq_linear",
            torch::nn::Sequential(
                torch::nn::Flatten(), layer_init(torch::nn::Linear(n_flatten, 512)), torch::nn::Tanh(), layer_init(torch::nn::Linear(512, 1))));
    }

    torch::Tensor forward(const torch::Tensor& x)
    {
        auto cnn_features = seq_cnn->forward(x);
        return seq_linear->forward(cnn_features);
    }

    torch::Tensor get_value(torch::Tensor x) { return forward(x); }

    torch::nn::Sequential seq_cnn{nullptr};
    torch::nn::Sequential seq_linear{nullptr};
};

TORCH_MODULE(CNNValueNetwork);
