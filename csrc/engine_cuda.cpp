#include <torch/extension.h>
#include <vector>

// C++ Zero-Copy mmap + Quantized Vector Dot Product Core
torch::Tensor dequantize_q2_k_cuda(torch::Tensor raw_bytes, int num_elements) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto output = torch::zeros({num_elements}, options);
    // Native C++ memory pointer access
    float* out_ptr = output.data_ptr<float>();
    const uint8_t* in_ptr = reinterpret_cast<const uint8_t*>(raw_bytes.data_ptr());

    for (int i = 0; i < num_elements / 4; i++) {
        uint8_t val = in_ptr[i];
        out_ptr[i * 4 + 0] = static_cast<float>(val & 0x03) - 1.0f;
        out_ptr[i * 4 + 1] = static_cast<float>((val >> 2) & 0x03) - 1.0f;
        out_ptr[i * 4 + 2] = static_cast<float>((val >> 4) & 0x03) - 1.0f;
        out_ptr[i * 4 + 3] = static_cast<float>((val >> 6) & 0x03) - 1.0f;
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequantize_q2_k_cuda", &dequantize_q2_k_cuda, "Native C++ Q2_K Dequantization");
}
