// Bindings of mx4_rocm.cu (RDNA4's MXFP4 matmuls; credits in that file).
#include <torch/extension.h>

void prompt_mx4(const at::Tensor& x8, const at::Tensor& a, const at::Tensor& wt, const at::Tensor& sct,
                const at::Tensor& ref, int n, at::Tensor& out, int tn);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prompt", &prompt_mx4, "FP8 prompt fragments times MXFP4 weights, the scale folded into e4m3");
}
