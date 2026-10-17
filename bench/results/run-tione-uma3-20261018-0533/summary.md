
### uma m, 300 atoms (uma)

| method | ms/step | peak GiB | max atoms | atoms/s at max |
|---|---|---|---|---|
| cg | 240.1 | 27.01 | - | - |
| cg_budget | 246.6 | 24.88 | - | - |
| cg_ckpt | 445.9 | 3.82 | - | - |
| cg_eager | 420.5 | 26.72 | - | - |

| create_graph | vs | speedup | memory saved | max batch |
|---|---|---|---|---|
| cg | cg_eager | 1.75x | -1% | - |
| cg_budget | cg_eager | 1.71x | +7% | - |
| cg_ckpt | cg_eager | 0.94x | +86% | - |

### uma s, 300 atoms (uma)

| method | ms/step | peak GiB | max atoms | atoms/s at max |
|---|---|---|---|---|
| cg | 51.5 | 5.02 | - | - |
| cg_budget | 50.5 | 4.85 | - | - |
| cg_ckpt | 82.0 | 1.64 | - | - |
| cg_eager | 90.4 | 5.30 | - | - |

| create_graph | vs | speedup | memory saved | max batch |
|---|---|---|---|---|
| cg | cg_eager | 1.75x | +5% | - |
| cg_budget | cg_eager | 1.79x | +8% | - |
| cg_ckpt | cg_eager | 1.10x | +69% | - |
