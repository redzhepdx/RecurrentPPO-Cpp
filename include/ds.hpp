#pragma once
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "stb_image_write.h"

template <typename T> class CircularBuffer {
    std::vector<T> buf;
    int64_t head = 0, tail = 0;
    size_t capacity;
    bool full = false;

  public:
    CircularBuffer(size_t cap) : buf(cap), capacity(cap) {}

    void push(T value)
    {
        buf[head] = value;
        head      = (head + 1) % capacity;
        if (full)
            tail = (tail + 1) % capacity;
        full = head == tail;
    }

    bool empty() const { return (!full && head == tail); }

    bool is_full() const { return full; }

    size_t size() { return capacity; }

    T get(size_t index) { return buf[index]; }

    size_t pop_first()
    {
        if (empty()) {
            throw std::runtime_error("pop_first on empty buffer");
        }
        T val = buf[tail];
        full  = false;
        tail  = (tail + 1) % capacity;
        return val;
    }

    T sum()
    {
        T total = (T)0;
        for (size_t i = 0; i < capacity; ++i) {
            total += buf[i];
        }
        return total;
    }
};

static void tensor_to_uint8(const torch::Tensor& t, std::vector<uint8_t>& out, int64_t& H, int64_t& W, int64_t& C)
{
    // Expect [C,H,W] or [H,W]
    torch::Tensor img = t.detach().cpu();

    if (img.dim() == 2) { // grayscale
        img = img.unsqueeze(0);
    }

    C = img.size(0);
    H = img.size(1);
    W = img.size(2);

    // Normalize to [0,1] then to [0,255]
    auto minv = img.min().item<float>();
    auto maxv = img.max().item<float>();
    if (maxv > minv) {
        img = (img - minv) / (maxv - minv);
    }

    img = (img * 255).clamp(0, 255).to(torch::kU8);

    out.resize(C * H * W);
    std::memcpy(out.data(), img.data_ptr(), out.size());
}

struct CircularTensorBuffer {
    torch::Tensor buffer;
    int64_t head      = 0;
    int64_t capacity_ = 5;
    int64_t channels_;
    int64_t height_;
    int64_t width_;

    CircularTensorBuffer(int64_t C, int64_t H, int64_t W, torch::Device device, int64_t capacity) : channels_(C), height_(H), width_(W)
    {
        capacity_ = capacity;
        if (C > 1) {
            buffer = torch::zeros({capacity + 1, C, H, W}, device);
        } else {
            buffer = torch::zeros({capacity + 1, H, W}, device);
        }
    }

    // x: [1, C, H, W] or [1, H, W]
    void push_front(const torch::Tensor& x)
    {
        buffer[head].copy_(x[0]);
        head = (head + 1) % (capacity_ + 1);
    }

    // returns [capacity_, C, H, W] or [capacity_, H, W] with newest at index 0
    torch::Tensor view() const
    {
        auto idx = (head - 1 - torch::arange(capacity_, buffer.device()) + (capacity_ + 1)) % (capacity_ + 1);
        return buffer.index_select(0, idx);
    }

    void reset()
    {
        buffer.zero_();
        head = 0;
    }

    void save_images(const std::string& dir)
    {
        std::filesystem::create_directories(dir);

        auto history      = this->view();
        int64_t hist_size = history.size(0);

        for (int64_t i = 0; i < hist_size; ++i) {
            torch::Tensor img = history[i];

            std::vector<uint8_t> buffer;
            int64_t H, W, C;
            tensor_to_uint8(img, buffer, H, W, C);

            std::ostringstream ss;
            ss << dir << "/img_" << std::setw(3) << std::setfill('0') << i << ".png";
            std::string filename = ss.str();

            // Write as PNG
            // stb wants channels last, so we pass H, W, C, and the data pointer
            stbi_write_png(filename.c_str(), (int)W, (int)H, (int)C, buffer.data(), (int)(W * C));
        }
    }
};