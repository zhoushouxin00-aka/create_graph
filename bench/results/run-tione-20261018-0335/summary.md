
### allscaip h640_l10, 1024 atoms (allscaip_2f2b)

| method | ms/step | peak GiB | max atoms | atoms/s at max |
|---|---|---|---|---|
| 2f2b | 915.2 | 45.02 | 1656 | 1133 |
| 2f2b_budget | 935.1 | 27.47 | 2485 | 1023 |
| 2f2b_ckpt | 995.0 | 20.60 | 3456 | 878 |
| cg | 666.3 | 47.34 | 1575 | 1545 |
| cg_budget | 709.5 | 40.82 | 1787 | 1480 |
| cg_ckpt | 1068.7 | 8.16 | 7391 | 784 |
| cg_eager | 1236.7 | 47.73 | 1536 | 797 |

| create_graph | vs | speedup | memory saved | max batch |
|---|---|---|---|---|
| cg | cg_eager | 1.86x | +1% | 1.03x |
| cg | 2f2b | 1.37x | -5% | 0.95x |
| cg_budget | 2f2b_budget | 1.32x | -49% | 0.72x |
| cg_ckpt | 2f2b_ckpt | 0.93x | +60% | 2.14x |

### allscaip h640_l10, 1024 atoms (allscaip_explicit)

| method | ms/step | peak GiB | max atoms | atoms/s at max |
|---|---|---|---|---|
| cg | 658.6 | 47.34 | 1575 | 1584 |
| cg_budget | 710.6 | 40.82 | 1787 | 1459 |
| cg_ckpt | 1054.8 | 8.16 | 7391 | 794 |
| cg_eager | 1234.0 | 47.73 | 1536 | 800 |
| explicit_budget | 734.4 | 36.57 | 1928 | 1269 |
| explicit_ckpt | 1077.1 | 8.63 | 7975 | 720 |
| explicit_compiled | 745.6 | 57.53 | 1285 | 1433 |

| create_graph | vs | speedup | memory saved | max batch |
|---|---|---|---|---|
| cg | cg_eager | 1.87x | +1% | 1.03x |
| cg | explicit_compiled | 1.13x | +18% | 1.23x |
| cg_budget | explicit_budget | 1.03x | -12% | 0.93x |
| cg_ckpt | explicit_ckpt | 1.02x | +5% | 0.93x |

### allscaip h640_l10, 1024 atoms (allscaip_explicit, budget0.3)

| method | ms/step | peak GiB | max atoms | atoms/s at max |
|---|---|---|---|---|
| cg_budget | 741.6 | 43.66 | - | - |
| explicit_budget | 732.7 | 37.02 | - | - |

| create_graph | vs | speedup | memory saved | max batch |
|---|---|---|---|---|
| cg_budget | explicit_budget | 0.99x | -18% | - |

### allscaip h640_l10, 1024 atoms (allscaip_explicit, budget0.7)

| method | ms/step | peak GiB | max atoms | atoms/s at max |
|---|---|---|---|---|
| cg_budget | 681.7 | 38.11 | - | - |
| explicit_budget | 678.8 | 36.57 | - | - |

| create_graph | vs | speedup | memory saved | max batch |
|---|---|---|---|---|
| cg_budget | explicit_budget | 1.00x | -4% | - |

### allscaip h640_l10, 1536 atoms (allscaip_explicit_large)

| method | ms/step | peak GiB | max atoms | atoms/s at max |
|---|---|---|---|---|
| cg_budget | 1146.9 | 62.56 | - | - |
| cg_ckpt | 1645.1 | 12.31 | - | - |
| explicit_budget | 1121.7 | 56.09 | - | - |
| explicit_ckpt | 1692.4 | 12.99 | - | - |

| create_graph | vs | speedup | memory saved | max batch |
|---|---|---|---|---|
| cg_budget | explicit_budget | 0.98x | -12% | - |
| cg_ckpt | explicit_ckpt | 1.03x | +5% | - |

### allscaip h640_l10, 2048 atoms (allscaip_explicit_large)

| method | ms/step | peak GiB | max atoms | atoms/s at max |
|---|---|---|---|---|
| cg_ckpt | 2216.6 | 16.45 | - | - |
| explicit_ckpt | 2244.6 | 17.32 | - | - |

| create_graph | vs | speedup | memory saved | max batch |
|---|---|---|---|---|
| cg_ckpt | explicit_ckpt | 1.01x | +5% | - |

### allscaip h640_l10, 3072 atoms (allscaip_explicit_large)

| method | ms/step | peak GiB | max atoms | atoms/s at max |
|---|---|---|---|---|
| cg_ckpt | 3396.3 | 25.03 | - | - |
| explicit_ckpt | 3469.0 | 26.16 | - | - |

| create_graph | vs | speedup | memory saved | max batch |
|---|---|---|---|---|
| cg_ckpt | explicit_ckpt | 1.02x | +4% | - |

### allscaip h128_l2, 1024 atoms (allscaip_sizes)

| method | ms/step | peak GiB | max atoms | atoms/s at max |
|---|---|---|---|---|
| 2f2b | 214.0 | 2.42 | 11664 | 6249 |
| cg | 186.4 | 3.44 | 11664 | 6991 |
| cg_eager | 191.4 | 3.34 | 11664 | 6233 |

| create_graph | vs | speedup | memory saved | max batch |
|---|---|---|---|---|
| cg | cg_eager | 1.03x | -3% | 1.00x |
| cg | 2f2b | 1.15x | -42% | 1.00x |

### allscaip h256_l4, 1024 atoms (allscaip_sizes)

| method | ms/step | peak GiB | max atoms | atoms/s at max |
|---|---|---|---|---|
| 2f2b | 270.8 | 8.35 | 6849 | 3559 |
| cg | 266.5 | 9.94 | 5593 | 4523 |
| cg_eager | 279.6 | 9.45 | 6035 | 3489 |

| create_graph | vs | speedup | memory saved | max batch |
|---|---|---|---|---|
| cg | cg_eager | 1.05x | -5% | 0.93x |
| cg | 2f2b | 1.02x | -19% | 0.82x |

### allscaip h512_l6, 1024 atoms (allscaip_sizes)

| method | ms/step | peak GiB | max atoms | atoms/s at max |
|---|---|---|---|---|
| 2f2b | 468.0 | 22.11 | 3122 | 2191 |
| cg | 387.3 | 24.03 | 2893 | 2805 |
| cg_eager | 508.9 | 23.84 | 3043 | 1909 |

| create_graph | vs | speedup | memory saved | max batch |
|---|---|---|---|---|
| cg | cg_eager | 1.31x | -1% | 0.95x |
| cg | 2f2b | 1.21x | -9% | 0.93x |

### nequip L, 2048 atoms (nequip)

| method | ms/step | peak GiB | max atoms | atoms/s at max |
|---|---|---|---|---|
| cg | 318.0 | 23.04 | 6570 | 5923 |
| cg_budget | 362.4 | 35.11 | 4725 | 5617 |
| cg_ckpt | 603.0 | 8.53 | 18570 | 2955 |
| cg_eager | 942.8 | 20.90 | 7649 | 2013 |
| nequip_compile | 285.1 | 23.05 | 6570 | 6516 |
| nequip_eager | 946.3 | 20.91 | 7649 | 2005 |

| create_graph | vs | speedup | memory saved | max batch |
|---|---|---|---|---|
| cg | nequip_eager | 2.98x | -10% | 0.86x |
| cg | nequip_compile | 0.90x | +0% | 1.00x |
| cg | cg_eager | 2.97x | -10% | 0.86x |
| cg_budget | nequip_compile | 0.79x | -52% | 0.72x |
| cg_ckpt | nequip_compile | 0.47x | +63% | 2.83x |

### nequip M, 4096 atoms (nequip)

| method | ms/step | peak GiB | max atoms | atoms/s at max |
|---|---|---|---|---|
| cg | 158.3 | 11.76 | 26047 | 19947 |
| cg_budget | 146.1 | 10.44 | 28826 | 25758 |
| cg_ckpt | 248.9 | 5.89 | 51633 | 13954 |
| cg_eager | 349.6 | 11.26 | 26716 | 11298 |
| nequip_compile | 149.8 | 10.92 | 27401 | 24751 |
| nequip_eager | 339.4 | 11.26 | 26716 | 11327 |

| create_graph | vs | speedup | memory saved | max batch |
|---|---|---|---|---|
| cg | nequip_eager | 2.14x | -4% | 0.97x |
| cg | nequip_compile | 0.95x | -8% | 0.95x |
| cg | cg_eager | 2.21x | -4% | 0.97x |
| cg_budget | nequip_compile | 1.03x | +4% | 1.05x |
| cg_ckpt | nequip_compile | 0.60x | +46% | 1.88x |

### nequip S, 4096 atoms (nequip)

| method | ms/step | peak GiB | max atoms | atoms/s at max |
|---|---|---|---|---|
| cg | 15.0 | 1.76 | 104976 | 347518 |
| cg_budget | 14.3 | 1.12 | 104976 | 347681 |
| cg_ckpt | 21.5 | 1.37 | 104976 | 218206 |
| cg_eager | 28.5 | 1.87 | 104976 | 153250 |
| nequip_compile | 24.6 | 1.81 | 104976 | 357707 |
| nequip_eager | 30.7 | 1.88 | 104976 | 101546 |

| create_graph | vs | speedup | memory saved | max batch |
|---|---|---|---|---|
| cg | nequip_eager | 2.05x | +7% | 1.00x |
| cg | nequip_compile | 1.64x | +3% | 1.00x |
| cg | cg_eager | 1.90x | +6% | 1.00x |
| cg_budget | nequip_compile | 1.72x | +38% | 1.00x |
| cg_ckpt | nequip_compile | 1.14x | +25% | 1.00x |

### pet default, 4096 atoms (pet)

| method | ms/step | peak GiB | max atoms | atoms/s at max |
|---|---|---|---|---|
| cg | 335.4 | 31.34 | 10199 | 12425 |
| cg_budget | 297.4 | 22.77 | 11576 | 11708 |
| cg_ckpt | 483.3 | 19.96 | 12811 | 7440 |
| cg_eager | 498.5 | 33.75 | 8540 | 8001 |
| native_eager | 518.4 | 33.75 | 8540 | 7015 |

| create_graph | vs | speedup | memory saved | max batch |
|---|---|---|---|---|
| cg | native_eager | 1.55x | +7% | 1.19x |
| cg | cg_eager | 1.49x | +7% | 1.19x |
| cg_budget | cg_eager | 1.68x | +33% | 1.36x |
| cg_ckpt | cg_eager | 1.03x | +41% | 1.50x |

### pet large, 2048 atoms (pet)

| method | ms/step | peak GiB | max atoms | atoms/s at max |
|---|---|---|---|---|
| cg | 387.1 | 38.72 | 3858 | 5333 |
| cg_budget | 356.5 | 27.40 | 5364 | 5573 |
| cg_ckpt | 567.5 | 19.65 | 7271 | 3213 |
| cg_eager | 578.4 | 41.51 | 3762 | 3477 |
| native_eager | 678.4 | 41.51 | 3762 | 2920 |

| create_graph | vs | speedup | memory saved | max batch |
|---|---|---|---|---|
| cg | native_eager | 1.75x | +7% | 1.03x |
| cg | cg_eager | 1.49x | +7% | 1.03x |
| cg_budget | cg_eager | 1.62x | +34% | 1.43x |
| cg_ckpt | cg_eager | 1.02x | +53% | 1.93x |

### uma m, 300 atoms (uma)

| method | ms/step | peak GiB | max atoms | atoms/s at max |
|---|---|---|---|---|
| cg | 241.6 | 27.01 | - | - |
| cg_budget | 244.4 | 24.88 | - | - |
| cg_ckpt | 2670.6 | 3.82 | - | - |
| cg_eager | 429.7 | 26.72 | - | - |

| create_graph | vs | speedup | memory saved | max batch |
|---|---|---|---|---|
| cg | cg_eager | 1.78x | -1% | - |
| cg_budget | cg_eager | 1.76x | +7% | - |
| cg_ckpt | cg_eager | 0.16x | +86% | - |

### uma s, 300 atoms (uma)

| method | ms/step | peak GiB | max atoms | atoms/s at max |
|---|---|---|---|---|
| cg | 65.3 | 5.02 | - | - |
| cg_budget | 56.4 | 4.85 | - | - |
| cg_ckpt | 2255.4 | 1.64 | - | - |
| cg_eager | 126.2 | 5.30 | - | - |

| create_graph | vs | speedup | memory saved | max batch |
|---|---|---|---|---|
| cg | cg_eager | 1.93x | +5% | - |
| cg_budget | cg_eager | 2.24x | +8% | - |
| cg_ckpt | cg_eager | 0.06x | +69% | - |
