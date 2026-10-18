# DC-Ops E8 overnight report

Updated 2026-10-18 01:06

## Vision students (Ethos-U85 exports)

| run | val acc | board-sim acc | int8 agrees w/ float | .pte KB | CPU ops |
|---|---|---|---|---|---|
| r128_kd | 0.581 | 0.560 | 1.0 | 2302 | cortex_m::dequantize_per_tensor, cortex_m::quantize_per_tensor |
| r160_kd | 0.653 | 0.593 | 0.984375 | 2303 | cortex_m::dequantize_per_tensor, cortex_m::quantize_per_tensor |
| r160_kd_e120 | 0.644 | 0.636 | 0.953125 | 2306 | cortex_m::dequantize_per_tensor, cortex_m::quantize_per_tensor |
| r160_noaug | 0.584 | 0.498 | 0.953125 | 2301 | cortex_m::dequantize_per_tensor, cortex_m::quantize_per_tensor |
| r160_nokd | 0.667 | 0.536 | 0.953125 | 2304 | cortex_m::dequantize_per_tensor, cortex_m::quantize_per_tensor |
| r192_kd | 0.632 | 0.531 | 0.96875 | 2301 | cortex_m::dequantize_per_tensor, cortex_m::quantize_per_tensor |
| v2_r160_kd | 0.912 | 0.913 | 1.0 | 2302 | cortex_m::dequantize_per_tensor, cortex_m::quantize_per_tensor |
| v2_r160_kd_mix | 0.914 | 0.814 | 0.984375 | 2301 | cortex_m::dequantize_per_tensor, cortex_m::quantize_per_tensor |
| v2_r160_kd_mix_e80 | 0.927 | 0.839 | 1.0 | 2304 | cortex_m::dequantize_per_tensor, cortex_m::quantize_per_tensor |
| v2_r160_noaug | 0.923 | 0.758 | 1.0 | 2295 | cortex_m::dequantize_per_tensor, cortex_m::quantize_per_tensor |
| v2_r160_nokd | 0.912 | 0.804 | 1.0 | 2304 | cortex_m::dequantize_per_tensor, cortex_m::quantize_per_tensor |

## Keyword spotters

| run | test acc | params | int8 agrees w/ float | .pte KB |
|---|---|---|---|---|
| dscnn_m_agc | 0.9466 | 79116 | 0.8828125 | 93 |
| dscnn_m_kd | 0.9444 | 79116 | 0.890625 | 93 |
| dscnn_s_agc | 0.9394 | 23180 | 0.88671875 | 34 |
| dscnn_s_kd | 0.9366 | 23180 | 0.8828125 | 33 |
