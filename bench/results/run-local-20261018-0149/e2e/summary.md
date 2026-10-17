| run | atoms/s (steps 10-29) | peak GiB (nvidia-smi) | steps logged |
|---|---|---|---|
| 2f2b_1024 | 1137 | 45.5 | 30 |
| 2f2b_1536 | 853 | 46.3 | 30 |
| cons_1024 | 1020 | 48.4 | 30 |
| explicit_1024 | 1463 | 50.8 | 30 |
| explicit_1536 | 1374 | 51.9 | 30 |
| generated_1024 | 1461 | 48.3 | 30 |
| generated_1536 | 1378 | 64.4 | 30 |
| generated_ckpt_1024 | 912 | 22.1 | 30 |

2 GPUs, 4 steps:

- generated: steps [0, 1, 2, 3], 439 parameter gradients, ranks identical: True
- cons: steps [0, 1, 2, 3], 439 parameter gradients, ranks identical: True
- rank 0 step 0: generated vs create_graph loss rel 9.3e-06, worst grad abs-mean rel 1.3e-03 over 403 parameters (36 at rounding level, below 1e-6 of the largest)
- rank 0 step 1: generated vs create_graph loss rel 5.3e-06, worst grad abs-mean rel 5.1e-04 over 403 parameters (36 at rounding level, below 1e-6 of the largest)
- rank 0 step 2: generated vs create_graph loss rel 4.5e-06, worst grad abs-mean rel 1.1e-03 over 390 parameters (49 at rounding level, below 1e-6 of the largest)
- rank 0 step 3: generated vs create_graph loss rel 9.9e-06, worst grad abs-mean rel 9.8e-04 over 388 parameters (51 at rounding level, below 1e-6 of the largest)
- rank 1 step 0: generated vs create_graph loss rel 1.8e-05, worst grad abs-mean rel 1.3e-03 over 403 parameters (36 at rounding level, below 1e-6 of the largest)
- rank 1 step 1: generated vs create_graph loss rel 8.2e-06, worst grad abs-mean rel 5.1e-04 over 403 parameters (36 at rounding level, below 1e-6 of the largest)
- rank 1 step 2: generated vs create_graph loss rel 1.2e-05, worst grad abs-mean rel 1.1e-03 over 390 parameters (49 at rounding level, below 1e-6 of the largest)
- rank 1 step 3: generated vs create_graph loss rel 1.2e-05, worst grad abs-mean rel 9.8e-04 over 388 parameters (51 at rounding level, below 1e-6 of the largest)
