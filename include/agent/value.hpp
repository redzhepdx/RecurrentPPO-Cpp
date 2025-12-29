#pragma once

#include <networks.hpp>>
#include <torch/torch.h>

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

        seq_linear = register_module(
            "seq_linear",
            torch::nn::Sequential(
                torch::nn::Flatten(), layer_init(torch::nn::Linear(128 * 6 * 6, 512)), torch::nn::Tanh(), layer_init(torch::nn::Linear(512, 1))));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        auto cnn_features = seq_cnn->forward(x);
        return seq_linear->forward(cnn_features);
    }

    torch::Tensor get_value(torch::Tensor x) { return forward(x); }

    torch::nn::Sequential seq_cnn{nullptr};
    torch::nn::Sequential seq_linear{nullptr};
};

TORCH_MODULE(CNNValueNetwork);
