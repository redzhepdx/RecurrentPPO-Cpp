#pragma once
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <torch/torch.h>

// Adapted from here https://github.com/adepierre/torchRL/blob/master/torchrl/src/rl/NormalDistribution.cpp
const static float half_log_2_pi = 0.5f * std::log(2.0f * M_PI);

class NormalDistribution {
  private:
    int64_t event_dim;
    torch::Tensor mean;
    torch::Tensor std;

  public:
    NormalDistribution(const torch::Tensor& mean_, const torch::Tensor& std_)
    {
        event_dim = mean_.size(-1);
        mean      = mean_;
        std       = std_;
    }
    ~NormalDistribution() = default;

    torch::Tensor sample(const int64_t N)
    {
        torch::Tensor eps = torch::normal(0.0, 1.0, {N, event_dim});
        return mean + eps * std;
    }
    torch::Tensor log_prob(const torch::Tensor& samples)
    {
        auto elementwise_log_output = (-std.log() - half_log_2_pi - torch::pow(samples - mean, 2) / (2.0f * torch::pow(std, 2)));
        return elementwise_log_output.sum(1, true);
    }
    torch::Tensor entropy() { return (0.5f + half_log_2_pi + torch::log(std)).sum(-1, true); }
};

struct MultiBernoulliDist {
    torch::Tensor logits; // shape [batch, n_actions]

    MultiBernoulliDist(torch::Tensor logits_) : logits(logits_) {}

    torch::Tensor sample()
    {
        // sample from Bernoulli using logit -> prob
        auto probs = torch::sigmoid(logits);
        return torch::bernoulli(probs);
    }

    torch::Tensor log_prob(const torch::Tensor& actions)
    {
        // log p(a) = sum_i [ a_i * log σ + (1-a_i)*log(1-σ) ]
        auto probs = torch::sigmoid(logits);
        auto logp1 = torch::log(probs + 1e-8);
        auto logp0 = torch::log(1.0 - probs + 1e-8);
        return (actions * logp1 + (1.0 - actions) * logp0).sum(-1, true);
    }

    torch::Tensor entropy()
    {
        auto probs = torch::sigmoid(logits);
        return (-probs * torch::log(probs + 1e-8) - (1.0 - probs) * torch::log(1.0 - probs + 1e-8)).sum(-1, true);
    }
};

struct CategoricalDist {
    torch::Tensor logits; // [batch, n_actions]

    CategoricalDist(torch::Tensor l) : logits(l) {}

    torch::Tensor sample()
    {
        auto probs = torch::softmax(logits, -1);
        return probs.multinomial(1); // [batch, 1]
    }

    torch::Tensor log_prob(const torch::Tensor& actions)
    {
        auto logp = torch::log_softmax(logits, -1);
        return logp.gather(1, actions);
    }

    torch::Tensor entropy()
    {
        auto p = torch::softmax(logits, -1);
        return (-p * torch::log(p + 1e-8)).sum(-1, true);
    }
};

static float EPS           = 1e-6f;
static float LOG_2_PI      = std::log(2.0f * M_PI);
static float HALF_LOG_2_PI = 0.5f * LOG_2_PI;

class TanhNormal {
  private:
    torch::Tensor mean;
    torch::Tensor std;
    int64_t event_dim;

  public:
    TanhNormal(const torch::Tensor& mean_, const torch::Tensor& std_) : mean(mean_), std(std_), event_dim(mean_.size(-1)) {}

    ~TanhNormal() = default;

    // Sample action (returns squashed action and raw)
    std::tuple<torch::Tensor, torch::Tensor> sample(int64_t N)
    {
        auto eps = torch::normal(0.0, 1.0, {N, event_dim}).to(mean.dtype());
        auto raw = mean + eps * std;
        auto act = torch::tanh(raw);
        return {act, raw};
    }

    // Log probability after tanh
    torch::Tensor log_prob(const torch::Tensor& actions)
    {
        auto clipped = actions.clamp(-1.0 + EPS, 1.0 - EPS);
        auto raw     = 0.5f * (torch::log1p(clipped) - torch::log1p(-clipped));

        auto log_p_raw = -HALF_LOG_2_PI - std.log() - (raw - mean).pow(2).div(2.0f * std.pow(2));
        log_p_raw      = log_p_raw.sum(-1, true);

        auto log_det = (1.0f - clipped.pow(2) + EPS).log().sum(-1, true);

        return log_p_raw - log_det;
    }

    // Base normal entropy
    torch::Tensor entropy() { return (0.5f + HALF_LOG_2_PI + std.log()).sum(-1, true); }
};

std::vector<std::string> split(const std::string& s, const std::string& delim)
{
    std::vector<std::string> tokens;
    std::string str = s; // make a copy if you want to modify it

    size_t pos = 0;
    while ((pos = str.find(delim)) != std::string::npos) {
        tokens.push_back(str.substr(0, pos));
        str.erase(0, pos + delim.length());
    }
    tokens.push_back(str); // last part
    return tokens;
}