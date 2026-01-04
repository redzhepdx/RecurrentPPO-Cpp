#pragma once

#include <cmath>
#include <vector>

#include <torch/torch.h>

#include "distributions.hpp"
#include "networks.hpp"

struct CNNActorCriticNetworkImpl : torch::nn::Module {

    CNNActorCriticNetworkImpl() = default;

    CNNActorCriticNetworkImpl(std::tuple<size_t, size_t, size_t> obs_size, size_t action_dim)
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

        seq_actor = register_module("seq_feature",
                                    torch::nn::Sequential(torch::nn::Flatten(),
                                                          layer_init(torch::nn::Linear(n_flatten, 512)),
                                                          torch::nn::Tanh(),
                                                          layer_init(torch::nn::Linear(512, action_dim))));

        seq_value = register_module(
            "seq_value",
            torch::nn::Sequential(
                torch::nn::Flatten(), layer_init(torch::nn::Linear(n_flatten, 512)), torch::nn::Tanh(), layer_init(torch::nn::Linear(512, 1))));

        // nn.Parameter equivalent
        actor_log_std = register_parameter("actor_log_std", torch::full({static_cast<long long>(action_dim)}, -3.0));
    }

    torch::Tensor forward(const torch::Tensor& x)
    {
        auto cnn_features = seq_cnn->forward(x);
        return seq_actor->forward(cnn_features);
    }

    torch::Tensor get_value(const torch::Tensor& x)
    {
        auto cnn_features = seq_cnn->forward(x);
        return seq_value->forward(cnn_features);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> get_actions(const torch::Tensor& x,
                                                                                       c10::optional<torch::Tensor> action_opt = c10::nullopt)
    {
        auto cnn_features = seq_cnn->forward(x);
        auto action_means = seq_actor->forward(cnn_features);

        CategoricalDist dist(action_means);

        torch::Tensor action_out;
        if (action_opt.has_value()) {
            action_out = action_opt.value();
        } else {
            action_out = dist.sample();
        }
        auto log_prob = dist.log_prob(action_out);

        auto entropy = dist.entropy();

        auto value = seq_value->forward(cnn_features);

        return {action_out, log_prob, entropy, value};
    }

    torch::nn::Sequential seq_cnn   = nullptr;
    torch::nn::Sequential seq_actor = nullptr;
    torch::nn::Sequential seq_value = nullptr;
    torch::Tensor actor_log_std;
};

struct RNNActorCriticNetworkImpl : torch::nn::Module {

    RNNActorCriticNetworkImpl() = default;

    RNNActorCriticNetworkImpl(std::tuple<size_t, size_t, size_t> obs_size, size_t action_dim)
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

        seq_feature = register_module("seq_feature",
                                      torch::nn::Sequential(torch::nn::Flatten(), layer_init(torch::nn::Linear(n_flatten, 512)), torch::nn::ReLU()));

        rnn = register_module("rnn", torch::nn::LSTM(512, 128));

        // Safe LSTM initializaiton
        {
            torch::NoGradGuard no_grad;

            for (auto& named_param : rnn->named_parameters()) {
                const std::string& param_name = named_param.key();
                auto& param                   = named_param.value();
                if (param_name.find("bias") != std::string::npos) {
                    torch::nn::init::constant_(param, 0.0);
                } else if (param_name.find("weight") != std::string::npos) {
                    torch::nn::init::orthogonal_(param, 1.0);
                }
            }
        }

        seq_actor = register_module("seq_actor", torch::nn::Sequential(layer_init(torch::nn::Linear(128, action_dim))));

        seq_value = register_module("seq_value", torch::nn::Sequential(layer_init(torch::nn::Linear(128, 1))));
    }

    std::tuple<int64_t, int64_t> get_lstm_info()
    {
        auto num_layers  = rnn->options.num_layers();
        auto hidden_size = rnn->options.hidden_size();
        return {num_layers, hidden_size};
    }

    std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>>
    get_states(const torch::Tensor& x, const std::tuple<torch::Tensor, torch::Tensor>& lstm_state, torch::Tensor& done)
    {
        auto hidden = seq_feature->forward(seq_cnn->forward(x)); // [B, 512]

        auto [h_state, c_state] = lstm_state;

        // LSTM Logic
        auto batch_size = std::get<0>(lstm_state).size(1);
        hidden          = hidden.reshape({-1, batch_size, 512}); // [T, B, 512]

        if (done.ndimension() == 1) {
            done = done.unsqueeze(1);
        }

        auto done_r = done.reshape({-1, batch_size}); // [T, B]

        std::vector<torch::Tensor> outputs;
        outputs.reserve(hidden.size(0));

        for (int64_t t = 0; t < hidden.size(0); ++t) {
            auto h_t = hidden[t]; // [B, 512]
            auto d_t = done_r[t]; // [B]

            auto mask = (1.0 - d_t).view({1, -1, 1}); // [1, B, 1]

            // Masked states
            auto h0 = h_state * mask;
            auto c0 = c_state * mask;

            auto [out, new_state] = rnn->forward(h_t.unsqueeze(0), std::make_tuple(h0, c0));

            h_state = std::get<0>(new_state);
            c_state = std::get<1>(new_state);

            outputs.push_back(out); // [1, B, H]
        }

        // [T, B, H] → [T*B, H]
        auto output = torch::cat(outputs, 0).flatten(0, 1);

        return {output, std::make_tuple(h_state, c_state)};
    }

    torch::Tensor forward(const torch::Tensor& x, const std::tuple<torch::Tensor, torch::Tensor>& lstm_state, torch::Tensor& done)
    {
        auto res = get_actions(x, lstm_state, done);
        return std::get<0>(res);
    }

    torch::Tensor get_value(const torch::Tensor& x, const std::tuple<torch::Tensor, torch::Tensor>& lstm_state, torch::Tensor& done)
    {
        auto [out, _] = get_states(x, lstm_state, done);
        return seq_value->forward(out);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>>
    get_actions(const torch::Tensor& x,
                const std::tuple<torch::Tensor, torch::Tensor>& lstm_state,
                torch::Tensor& done,
                c10::optional<torch::Tensor> action_opt = c10::nullopt)
    {
        auto [out, lstm_state_new] = get_states(x, lstm_state, done);

        auto action_means = seq_actor->forward(out);

        CategoricalDist dist(action_means);

        torch::Tensor action_out;
        if (action_opt.has_value()) {
            action_out = action_opt.value(); //.unsqueeze(0);
        } else {
            action_out = dist.sample();
        }

        auto log_prob = dist.log_prob(action_out);

        auto entropy = dist.entropy();

        auto value = seq_value->forward(out);

        return {action_out, log_prob, entropy, value, lstm_state_new};
    }

    torch::nn::Sequential seq_cnn     = nullptr;
    torch::nn::Sequential seq_feature = nullptr;
    torch::nn::Sequential seq_actor   = nullptr;
    torch::nn::Sequential seq_value   = nullptr;
    torch::nn::LSTM rnn               = nullptr;
    torch::Tensor actor_log_std;
};

TORCH_MODULE(CNNActorCriticNetwork);
TORCH_MODULE(RNNActorCriticNetwork);
