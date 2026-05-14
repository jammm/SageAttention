/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0.
 */

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "attn_gfx12_native.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("qk_int8_sv_f16_d64_native_attn", &qk_int8_sv_f16_d64_native_attn_gfx12,
        py::arg("query"), py::arg("key"), py::arg("value"), py::arg("output"),
        py::arg("query_scale"), py::arg("key_scale"), py::arg("tensor_layout"),
        py::arg("is_causal"), py::arg("sm_scale"), py::arg("valid_kv_len") = 0);
  m.def("qk_rawq_int8_sv_f8_native_attn", &qk_rawq_int8_sv_f8_native_attn_gfx12,
        py::arg("query"), py::arg("key"), py::arg("value"), py::arg("output"),
        py::arg("key_scale"), py::arg("tensor_layout"),
        py::arg("is_causal"), py::arg("sm_scale"), py::arg("valid_kv_len") = 0);
  m.def("qk_int8_sv_f16_d64_prepare_attn_hnd", &qk_int8_sv_f16_d64_prepare_attn_hnd_gfx12,
        py::arg("query"), py::arg("key"), py::arg("value"), py::arg("is_causal"),
        py::arg("value_is_fp8"), py::arg("use_raw_f16_value"), py::arg("sm_scale"),
        py::arg("valid_kv_len") = 0);
  m.def("quant_q_nhd_per_warp", &quant_q_nhd_per_warp_gfx12);
  m.def("transpose_value_fp8_hnd", &transpose_value_fp8_hnd_gfx12);
  m.def("transpose_value_f16_hnd", &transpose_value_f16_hnd_gfx12);
  m.def("convert_f16_to_bf16", &convert_f16_to_bf16_gfx12);
}
