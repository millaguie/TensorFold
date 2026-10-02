// Bindings of qmm8_rocm.cu (RDNA4's fp8 WMMA prompt matmul; credits in that file).
#include <torch/extension.h>

void gemm8(const at::Tensor& x8, const at::Tensor& xs, const at::Tensor& a, const at::Tensor& w8,
           const at::Tensor& scales, const at::Tensor& biases, int n, int group, at::Tensor& out, int variant);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm8", &gemm8, "RDNA4 FP8 prompt matmul on group-major scales");
}
