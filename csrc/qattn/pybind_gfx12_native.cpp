/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0.
 */

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "attn_gfx12_native.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("qk_int8_sv_f16_d64_native_attn", &qk_int8_sv_f16_d64_native_attn_gfx12);
  m.def("qk_int8_sv_f16_d64_prepare_attn_hnd", &qk_int8_sv_f16_d64_prepare_attn_hnd_gfx12);
  m.def("transpose_value_fp8_hnd", &transpose_value_fp8_hnd_gfx12);
  m.def("transpose_value_f16_hnd", &transpose_value_f16_hnd_gfx12);
  m.def("convert_f16_to_bf16", &convert_f16_to_bf16_gfx12);
  m.def("quant_qk_int8_hnd", &quant_qk_int8_hnd_gfx12);
  m.def("prepare_qkv_f16_hnd", &prepare_qkv_f16_hnd_gfx12);
  m.def("prepare_qkv_fp8_hnd", &prepare_qkv_fp8_hnd_gfx12);
}
