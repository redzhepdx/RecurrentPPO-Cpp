#pragma once

#include <torch/torch.h>

// Orthogonal init + constant bias (like your Python version)
template <typename Layer> Layer layer_init(Layer layer, double std = std::sqrt(2.0), double bias_const = 0.0)
{
    torch::nn::init::orthogonal_(layer->weight, std);
    if (layer->bias.defined()) {
        torch::nn::init::constant_(layer->bias, bias_const);
    }
    return layer;
}