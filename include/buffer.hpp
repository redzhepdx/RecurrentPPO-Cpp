#pragma once

#include <tuple>

#include <torch/torch.h>

static torch::Tensor allocate_with_capacity_like(const torch::Tensor& ref, size_t capacity)
{
    std::vector<int64_t> shape;
    shape.reserve(ref.dim() + 1);
    shape.push_back(static_cast<int64_t>(capacity));
    for (auto s : ref.sizes()) {
        shape.push_back(s);
    }
    return torch::empty(shape, ref.options());
}

class TrajectoryBuffer {
  private:
    int64_t capacity_;
    size_t position_ = 0;

    torch::Tensor states_;
    torch::Tensor actions_;
    torch::Tensor rewards_;
    torch::Tensor next_states_;
    torch::Tensor dones_;
    torch::Tensor terminates_;
    torch::Tensor truncates_;
    torch::Tensor log_probs_;
    torch::Tensor values_;
    torch::Tensor terminal_values_;

  public:
    TrajectoryBuffer(size_t capacity) : capacity_(capacity) {}

    void add(const torch::Tensor& state,
             const torch::Tensor& action,
             const torch::Tensor& log_prob,
             const torch::Tensor& reward,
             const torch::Tensor& next_state,
             const torch::Tensor& done,
             const torch::Tensor& terminated,
             const torch::Tensor& truncated,
             const torch::Tensor& value,
             const torch::Tensor& terminal_value)
    {

        if (!states_.defined() || states_.numel() == 0) {
            states_      = allocate_with_capacity_like(state, capacity_);
            next_states_ = allocate_with_capacity_like(next_state, capacity_);
            actions_     = allocate_with_capacity_like(action, capacity_);
            log_probs_   = allocate_with_capacity_like(log_prob, capacity_);

            rewards_         = torch::empty({static_cast<int64_t>(capacity_), 1}, reward.options());
            dones_           = torch::empty({static_cast<int64_t>(capacity_), 1}, done.options());
            terminates_      = torch::empty({static_cast<int64_t>(capacity_), 1}, done.options());
            truncates_       = torch::empty({static_cast<int64_t>(capacity_), 1}, done.options());
            values_          = torch::empty({static_cast<int64_t>(capacity_), 1}, value.options());
            terminal_values_ = torch::empty({static_cast<int64_t>(capacity_), 1}, value.options());
        }

        states_[position_]      = state;
        actions_[position_]     = action;
        log_probs_[position_]   = log_prob;
        rewards_[position_]     = reward;
        next_states_[position_] = next_state;
        dones_[position_]       = done;
        truncates_[position_]   = truncated;
        terminates_[position_]  = terminated;

        values_[position_]          = value;
        terminal_values_[position_] = terminal_value;

        position_ = (position_ + 1) % capacity_;
    };

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    sample(size_t batch_size)
    {
        auto indices = torch::randint(0, states_.size(0), {(long long)batch_size}, torch::kLong);

        return {states_.index_select(0, indices),
                actions_.index_select(0, indices),
                rewards_.index_select(0, indices),
                next_states_.index_select(0, indices),
                dones_.index_select(0, indices),
                terminates_.index_select(0, indices),
                truncates_.index_select(0, indices),
                values_.index_select(0, indices),
                terminal_values_.index_select(0, indices)};
    }

    torch::Tensor get_states() const { return states_.contiguous(); }
    torch::Tensor get_next_states() const { return next_states_.contiguous(); }
    torch::Tensor get_log_probs() const { return log_probs_.contiguous(); }
    torch::Tensor get_rewards() const { return rewards_.contiguous(); }
    torch::Tensor get_dones() const { return dones_.contiguous(); }
    torch::Tensor get_actions() const { return actions_.contiguous(); }
    torch::Tensor get_values() const { return values_.contiguous(); }
    torch::Tensor get_terminates() const { return terminates_.contiguous(); }
    torch::Tensor get_truncates() const { return truncates_.contiguous(); }
    torch::Tensor get_terminal_values() const { return terminal_values_.contiguous(); }

    size_t size() const { return states_.size(0); }

    void clear()
    {
        states_          = torch::empty({0});
        actions_         = torch::empty({0});
        rewards_         = torch::empty({0});
        next_states_     = torch::empty({0});
        dones_           = torch::empty({0});
        log_probs_       = torch::empty({0});
        values_          = torch::empty({0});
        terminates_      = torch::empty({0});
        truncates_       = torch::empty({0});
        terminal_values_ = torch::empty({0});
        position_        = 0;
    }
};

class ReplayBuffer {
  private:
    size_t capacity_;
    size_t size_     = 0;
    size_t position_ = 0;

    std::vector<TrajectoryBuffer> buffer_queue_;

  public:
    ReplayBuffer(size_t capacity) : capacity_(capacity) { buffer_queue_.reserve(capacity); }
    void add_trajectory(const TrajectoryBuffer& trajectory)
    {
        if (buffer_queue_.size() < capacity_) {
            buffer_queue_.push_back(trajectory);
            position_ = buffer_queue_.size() - 1;
        } else {
            position_                = (position_ + 1) % capacity_;
            buffer_queue_[position_] = trajectory;
        }

        if (size_ < capacity_) {
            size_++;
        }
    }

    size_t size() const { return size_; }

    void clear() { buffer_queue_.clear(); }

    const TrajectoryBuffer& get(size_t trajectory_idx) { return buffer_queue_[trajectory_idx]; }

    const TrajectoryBuffer& get_last() { return buffer_queue_.at(position_); }

    std::vector<std::reference_wrapper<const TrajectoryBuffer>> sample_tbs(size_t tb_count) const
    {
        TORCH_CHECK(tb_count <= buffer_queue_.size(), "tb_count too large");
        TORCH_CHECK(tb_count > 1, "tb_count too small");

        auto indices = torch::randint(0, buffer_queue_.size(), {static_cast<int64_t>(tb_count - 1)}, torch::kLong);

        std::vector<std::reference_wrapper<const TrajectoryBuffer>> batch;
        batch.reserve(tb_count);

        // Always include the last
        batch.emplace_back(buffer_queue_.back());

        for (int64_t i = 0; i < indices.size(0); ++i) {
            batch.emplace_back(buffer_queue_[indices[i].item<int64_t>()]);
        }

        return batch;
    }
};