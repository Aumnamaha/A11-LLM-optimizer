#include <torch/extension.h>
#include <cstdint>
#include <cstring>
#include <algorithm>

inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000) << 16;
    uint32_t exp  = (static_cast<uint32_t>(h) & 0x7C00) >> 10;
    uint32_t mant = (static_cast<uint32_t>(h) & 0x03FF);

    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FF;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(float));
    return out;
}

// Complete GGUF Q2_K dequantization implementation matching ggml format
torch::Tensor dequantize_q2_k_cuda(torch::Tensor raw_bytes, int num_elements) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto output = torch::zeros({num_elements}, options);
    if (num_elements <= 0 || raw_bytes.numel() == 0) {
        return output;
    }

    float* out_ptr = output.data_ptr<float>();
    const uint8_t* in_ptr = reinterpret_cast<const uint8_t*>(raw_bytes.data_ptr());
    int total_bytes = raw_bytes.numel();
    int num_blocks = total_bytes / 84;

    int out_idx = 0;
    for (int b = 0; b < num_blocks && out_idx < num_elements; ++b) {
        const uint8_t* blk = in_ptr + b * 84;
        const uint8_t* scales = blk;
        const uint8_t* qs = blk + 16;
        uint16_t d_raw, dmin_raw;
        std::memcpy(&d_raw, blk + 80, sizeof(uint16_t));
        std::memcpy(&dmin_raw, blk + 82, sizeof(uint16_t));
        float d = fp16_to_fp32(d_raw);
        float min = fp16_to_fp32(dmin_raw);

        const uint8_t* q = qs;
        int is = 0;

        for (int n = 0; n < 2; ++n) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc0 = scales[is++];
                float dl0 = d * static_cast<float>(sc0 & 0x0F);
                float ml0 = min * static_cast<float>(sc0 >> 4);
                for (int l = 0; l < 16; ++l) {
                    if (out_idx < num_elements) {
                        uint8_t val = (q[l] >> shift) & 3;
                        out_ptr[out_idx++] = dl0 * static_cast<float>(val) - ml0;
                    }
                }

                uint8_t sc1 = scales[is++];
                float dl1 = d * static_cast<float>(sc1 & 0x0F);
                float ml1 = min * static_cast<float>(sc1 >> 4);
                for (int l = 0; l < 16; ++l) {
                    if (out_idx < num_elements) {
                        uint8_t val = (q[l + 16] >> shift) & 3;
                        out_ptr[out_idx++] = dl1 * static_cast<float>(val) - ml1;
                    }
                }
                shift += 2;
            }
            q += 32;
        }
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequantize_q2_k_cuda", &dequantize_q2_k_cuda, "Native C++ GGUF Q2_K Dequantization");
}
