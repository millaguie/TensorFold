// Bindings of mx4_rocm.cu (RDNA4's MXFP4 matmuls; credits in that file).
#include <torch/extension.h>

void prompt_mx4(const at::Tensor& x8, const at::Tensor& a, const at::Tensor& wt, const at::Tensor& sct,
                const at::Tensor& ref, int n, at::Tensor& out, int tn);

void decode_mx4(const at::Tensor& x, const at::Tensor& wt, const at::Tensor& sct, const at::Tensor& ref, int n,
                at::Tensor& out, at::Tensor& part);
int decode_slices(int n, int kg);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode", &decode_mx4, "up to 32 bf16 rows times MXFP4 weights on bf16 WMMA, row-count invariant");
    m.def("decode_slices", &decode_slices, "K slices of an (n, kg groups) weight");
    m.def("prompt", &prompt_mx4, "FP8 prompt fragments times MXFP4 weights, the scale folded into e4m3");
}
