| run | atoms/s (steps 10-29) | peak GiB (nvidia-smi) | steps logged |
|---|---|---|---|
| 2f2b_1024 | 1169 | 45.5 | 30 |
| 2f2b_1536 | 1065 | 46.3 | 30 |
| cons_1024 | 978 | 48.4 | 30 |
| explicit_1024 | 1482 | 50.8 | 30 |
| explicit_1536 | 1375 | 51.9 | 30 |
| generated_1024 | 1451 | 48.3 | 30 |
| generated_1536 | 1388 | 64.5 | 30 |
| generated_ckpt_1024 | 917 | 22.1 | 30 |

2 GPUs, 4 steps:

- generated: steps [0, 1, 2, 3], 439 parameter gradients, ranks identical: True
- cons: steps [0, 1, 2, 3], 439 parameter gradients, ranks identical: True
- rank 0 step 0: generated vs create_graph: loss 5.1e-06, gradient abs-means 6.1e-05 (norm) over 409 parameters, worst 1.6e-03 over the 311 above 1e-4 of the largest (30 key biases with zero gradient left out)
- rank 0 step 1: generated vs create_graph: loss 6.1e-07, gradient abs-means 1.2e-04 (norm) over 409 parameters, worst 1.5e-03 over the 310 above 1e-4 of the largest (30 key biases with zero gradient left out)
- rank 0 step 2: generated vs create_graph: loss 1.6e-05, gradient abs-means 1.2e-04 (norm) over 409 parameters, worst 1.8e-03 over the 307 above 1e-4 of the largest (30 key biases with zero gradient left out)
- rank 0 step 3: generated vs create_graph: loss 5.8e-06, gradient abs-means 1.2e-04 (norm) over 409 parameters, worst 9.4e-04 over the 309 above 1e-4 of the largest (30 key biases with zero gradient left out)
- rank 1 step 0: generated vs create_graph: loss 1.8e-05, gradient abs-means 6.1e-05 (norm) over 409 parameters, worst 1.6e-03 over the 311 above 1e-4 of the largest (30 key biases with zero gradient left out)
- rank 1 step 1: generated vs create_graph: loss 5.3e-06, gradient abs-means 1.2e-04 (norm) over 409 parameters, worst 1.5e-03 over the 310 above 1e-4 of the largest (30 key biases with zero gradient left out)
- rank 1 step 2: generated vs create_graph: loss 9.7e-07, gradient abs-means 1.2e-04 (norm) over 409 parameters, worst 1.8e-03 over the 307 above 1e-4 of the largest (30 key biases with zero gradient left out)
- rank 1 step 3: generated vs create_graph: loss 9.1e-06, gradient abs-means 1.2e-04 (norm) over 409 parameters, worst 9.4e-04 over the 309 above 1e-4 of the largest (30 key biases with zero gradient left out)
