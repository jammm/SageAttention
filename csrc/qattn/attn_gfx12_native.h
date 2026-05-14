/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0.
 */

#pragma once

#include <torch/extension.h>

torch::Tensor qk_int8_sv_f16_d64_native_attn_gfx12(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor query_scale,
    torch::Tensor key_scale,
    int tensor_layout,
    int is_causal,
    float sm_scale,
    int64_t valid_kv_len = 0);

torch::Tensor qk_int8_sv_f16_d64_prepare_attn_hnd_gfx12(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    int is_causal,
    int value_is_fp8,
    int use_raw_f16_value,
    float sm_scale,
    int64_t valid_kv_len = 0);

torch::Tensor transpose_value_fp8_hnd_gfx12(torch::Tensor value);

torch::Tensor transpose_value_f16_hnd_gfx12(torch::Tensor value);

torch::Tensor convert_f16_to_bf16_gfx12(torch::Tensor input);
