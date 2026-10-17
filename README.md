# create_graph

Write the model once. `create_graph` generates the gradient of the energy with
respect to the positions into the forward graph of conservative-force MLIP
training. The force computation becomes an ordinary first-order graph:
AOTAutograd differentiates it, Inductor compiles it, and activation memory
budgets and per-block checkpointing apply to it, as if `dE/dx` had been written
by hand in the model's forward (fairchem `feature/explicit_create_graph` does
that by hand for AllScAIP). The VJP is PyTorch's own autograd formulas, traced
into the graph, so there is no rule library to maintain and no model code to
duplicate.

## API

Each kind of user touches one thing.

**The training step** asks for the energy and its gradient in one call:

```python
import create_graph as cg

out, grads = cg.create_graph(model, batch, outputs="energy", inputs=("pos",))
loss = loss_fn(out["energy"], -grads["pos"], targets)
loss.backward()                       # parameter gradient of the conservative-force loss
```

`cg.create_graph(model, data, *, outputs="energy", inputs=("pos",), train=None,
exact=False)` replaces `data.pos` (or `data["pos"]`) by a fresh leaf, calls
`model(data)` once and returns `(out, grads)`: the model output and, for every
name in `inputs`, the gradient of `sum(out[outputs])` with respect to it.
`outputs` is a key, an attribute, a callable selecting the tensor (fairchem's
`{"energy": {"energy": E}}` nesting), or None for a tensor output. With
`train=True` (the default when the model is training and grad is enabled) the
gradients are differentiable w.r.t. the parameters; `train=False` gives
first-order gradients and detached outputs (validation, MD). Under DDP pass the
DDP-wrapped model. `exact=True` runs eager `torch.autograd.grad(create_graph=True)`:
the reference the tests compare with.

**The model author** marks the part that computes the energy, typically the
parameterized network after the eager graph construction, with `cg.compile`, a
drop-in for `torch.compile`:

```python
self.network = cg.compile(self._network, config=cg.CompileConfig(memory_budget=0.5))
# or: model = cg.compile(model)   when the whole model is one graph
```

Outside `cg.create_graph` a region is `torch.compile(fn)`. Inside, it runs `fn`
and the VJP of its output as one compiled graph. `CompileConfig` fields:

| field | default | meaning |
|---|---|---|
| `backend` | `"inductor"` | `"aot_eager"` (no Inductor), `"eager"` (op by op, exact, for debugging) |
| `dynamic` | `True` | sizes symbolic from the first compile: one graph for every batch |
| `memory_budget` | `None` | AOTAutograd activation memory budget in (0, 1] for this region |
| `checkpoint` | `None` | regex (`re.fullmatch`) over the region's submodule paths: matching blocks are compiled one by one and recomputed in the backward |
| `options` | `()` | Inductor options for this region |
| `recompile_error` | `False` | raise on Dynamo's recompile limit instead of running eagerly |

**The kernel author** (only to speed up an op, or to make a non-traceable op
differentiable twice) registers forward1, backward1 and backward2, the backward
of backward1, in one class:

```python
@cg.custom_op("mylib::edge_attn")
class EdgeAttention:
    outputs = ("out", "lse")
    save = ("q", "k", "v", "bias", "lse")      # what forward1 keeps for backward1
    nondiff = ("lse",)                          # statistics: no gradient flows into them

    @staticmethod
    def forward1(q: Tensor, k: Tensor, v: Tensor, bias: Tensor, scale: float) -> tuple[Tensor, Tensor]: ...
    @staticmethod
    def backward1(q, k, v, bias, lse, grad_out, scale): ...        # -> grads of q, k, v, bias
    @staticmethod
    def backward2(q, k, v, bias, lse, grad_out, scale, gg_q, gg_k, gg_v, gg_bias): ...
        # -> grads of backward1's tensor inputs except statistics: q, k, v, bias, grad_out
    @staticmethod
    def fake(q, k, v, bias, scale): ...        # forward1's output shapes; backward fakes are generated
```

Under the hood these are three `torch.library.custom_op` operators and two
`register_autograd` calls, all public API, so ops registered with torch.library
directly work as well. `cg.testing.check_op(op, reference, args)` checks the
three kernels against a differentiable reference in float64.
[`kernels/edge_attn/`](kernels/edge_attn/ops.py) is the worked example (masked
attention with an additive bias; the analytic backward2 accounts for `lse`).

`cg.testing` also has `check_create_graph(model, data)` (one step against
`exact=True`: output, input gradients, a loss coupling both, all parameter
gradients) and `check_float32(...)` (a float32 model against its float64 copy,
next to eager create_graph's own float32 error).

## Integration contract

What a framework (fairchem, ...) may rely on within 0.1.x. Install create_graph
next to the framework without touching its dependencies:

```bash
uv pip install --no-deps -e /path/to/create_graph --python /path/to/framework/.venv/bin/python
```

**Stable API.** `create_graph`, `compile`, `custom_op`, `CompileConfig`,
`ContractError`, `FusedOp`, `reset`, `supports_in_graph_grad`,
`testing.check_create_graph`, `testing.check_float32`, `testing.check_op`.
Everything else is internal.

**Inside `cg.create_graph`**, breaking a rule raises `ContractError` with the
reason:

- exactly one region per call produces the differentiated output. Its VJP is
  seeded with ones (the gradient of the sum), so the model output selected by
  `outputs` must be the region's output, a view with the same entries
  (`squeeze`, `reshape`, unpadding slices), or that output plus terms that do not
  depend on the inputs (per-structure reference energies). Code after the region
  that scales entries or depends on the inputs again belongs in the region;
- eager code before the region (radius graph, edge vectors, spherical
  harmonics) may do anything, parameters included. create_graph chains the
  region's input gradients back to the positions with eager
  `torch.autograd.grad(create_graph=True)`;
- region inputs are followed through tuples, lists, dicts, namedtuples,
  dataclasses and the tensor attributes of plain objects (AtomicData). Inputs
  that depend only on the differentiated inputs become leaves; inputs that also
  depend on parameters are differentiated through; parameter-only inputs pass
  as they are;
- the model must not differentiate w.r.t. the inputs itself during the call
  (turn its force head off); other regions that do not depend on the inputs
  compile as usual, a second region that does is an error;
- a custom `torch.autograd.Function` on the path from the inputs to the energy
  is refused (checked once per region, eagerly): a compiled graph does not
  differentiate a traced Function's backward again, so its parameter gradients
  would silently be wrong (PyTorch 2.13). Register such an op with
  `cg.custom_op` or `torch.library` instead. Functions of parameters only are
  fine;
- with `checkpoint=`, Python scalars passed directly to a block start as
  compile-time constants (a layer index, a flag: one compiled forward and VJP
  per value, as if the region were compiled whole). One whose value changes
  between calls of the same block (UMA passes the number of atoms) is dynamic
  from then on, like Dynamo's automatic dynamic shapes. Scalars inside tensors
  and containers stay dynamic.

Without any region, `create_graph` warns once and returns eager create_graph's
result.

**Compilation.** A region is one graph (`fullgraph`). Sizes that happen to be
equal in the first batch (three structures and the xyz axis) share a symbol
under Dynamo's duck sizing, so the second batch may compile the region once
more; after that one compile covers every batch. Code that specializes on a
count (UMA on the number of structures per batch) compiles once per value.
Inside regions `F.rms_norm` is replaced by a composite (its fused backward
pins every size of the graph), DDPOptimizer is off (it would split the region
at bucket boundaries), and Inductor's mix-order reduction is off (on torch 2.11
its memory-coalescing analysis of tiling): both fail on dynamic `(K+1)*N`
ranges in backward graphs.

**Lifecycle.** `cg.reset()` forgets compiled regions and Dynamo's caches. Inductor's
FX graph cache does not key on a custom op's fake strides: after changing a
`fake`, clear `TORCHINDUCTOR_CACHE_DIR`.

**Versions.** The compiled path needs Dynamo to trace `torch.autograd.grad`
(`trace_autograd_ops`, torch >= 2.11). On an older torch (install with
`--no-deps`) `create_graph` warns once and runs eager create_graph: exact, not
compiled. Private torch APIs appear only in
[`compile_config.py`](src/create_graph/compile_config.py), behind feature
checks; `tests/test_torch_surface.py` enforces that.

## How it works

```
data.pos -> fresh leaf ─► eager geometry (autograd records it)
                             │ features enter detached
                             ▼
             cg.compile region: out = fn(...)
                                g = torch.autograd.grad(out.sum(), wrt, create_graph=train)
                             one torch.compile graph (Dynamo traces autograd.grad)
                             │
       out ◄─────────────────┤ g = dE/d(features)
                             ▼
             eager autograd.grad(features, pos, g, create_graph=True) ─► grads["pos"]

loss(out, grads).backward(): AOTAutograd's backward of the region graph (backward2)
```

The region body is `fn` followed by an in-graph `torch.autograd.grad`. Dynamo
captures both (`trace_autograd_ops`); AOTAutograd expands the inner backward
into ATen ops of the forward graph and derives the backward of the whole thing,
so the min-cut partitioner and the activation memory budget see the force
computation like any other forward code. `wrt` lists the region's inputs that
depend on the differentiated inputs, plus the parameter-only ones Dynamo needs
to see as inputs (their gradients are dropped).

Per-block checkpointing (`CompileConfig(checkpoint=...)`) runs the region
eagerly and turns every matching block into a `BlockFn` node: its forward is the
block's compiled forward without grad, keeping only the block's inputs. In the
force pass its backward is a `VjpFn` node that runs the block's compiled
forward + VJP; the parameter backward recomputes that forward + VJP with grad
(backward2) and the block forward for the first-order part. A block's forward
runs four times and its VJP twice, but only one block's activations are alive
at a time, and the input embedding, readout and glue between blocks need no
hand-written VJP.

## Correctness

Every case is compared with eager create_graph (`exact=True`): the output, the
input gradients, a loss coupling both, and every parameter gradient. Thresholds
(relative norm error per tensor): float64 1e-10 (aot_eager) and 1e-9
(Inductor); float32 without TF32 1e-5, or, for models where Inductor's float32
rounding exceeds that, no worse than eager create_graph against the float64
result (factor 4).

- `tests/test_semantics.py`, `test_custom_ops.py`, `test_torch_surface.py`
  (CPU, float64): eager geometry before the region, inputs of every kind,
  LayerNorm / RMSNorm / GELU / softmax, dropout masks shared by the forward and
  the VJP, `no_grad` / `detach` / in-place inside regions, Functions refused on
  the differentiated path, outputs and views, evaluation mode, structured
  inputs, every contract error, checkpointed blocks (with scalar arguments read
  in Python), one compile for five batch sizes, the eager fallback of old torch
  versions; the EdgeAttention and Cube fused operators against references.
- `tests/test_ddp.py`: two gloo ranks, gradients identical to single-process
  create_graph, the region stays one graph, and the DDPOptimizer regression.
- `tests/test_models_pet.py`, `test_models_nequip.py` (GPU, real metatrain PET
  and nequip NequIP on OMol25 batches): float32 for two PET sizes and three
  NequIP presets, float64, memory budget and checkpoint keep the gradients and
  lower the peak, periodic metals, the fused attention operator inside PET,
  nequip's own force output, one compile across batch sizes, one epoch in
  lockstep (float32) and free-running (float64).
- `tests/test_models_fairchem.py` (fairchem's venv): AllScAIP h128_l2 to
  h640_l10 (85M) in float32, memory budget and checkpoint, the EdgeAttention
  operator in AllScAIP, one compile across batches, one epoch in lockstep; UMA
  (eSCN-MD S and M) in float32, one compile per number of structures, and the
  Euler-angle path (fairchem's Safeacos / Safeatan2 Functions refused, accepted
  as `cg.custom_op` operators).
- `bench/ddp_parity.py` (torchrun, NCCL, float32) on PET and NequIP: 1x2, 1x4,
  1x8 GPUs here and 2 nodes x 8 GPUs on TI-ONE. Every rank holds bitwise
  identical reduced gradients, within 1.0e-7 (PET) and 1.5e-7 (NequIP) of eager
  create_graph through the same DDP wrapper, over 4 lockstep steps.

```bash
.venv/bin/python -m pytest tests -m "not gpu"                     # CPU suite
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest tests/test_models_pet.py tests/test_models_nequip.py
FAIRCHEM_ROOT=../fairchem-create_graph PYTHONPATH=../fairchem-create_graph/src:src:. \
  ../fairchem/.venv/bin/python -m pytest tests/test_models_fairchem.py
```

Pitfalls found on the way (all covered by tests):

- A compiled graph with in-graph `create_graph=True` does not differentiate a
  traced `torch.autograd.Function`'s backward again: the VJP through it is a
  constant for the parameter backward, and the parameter gradients come out
  wrong without any error. Hence the refusal above.
- On NequIP preset M, Inductor's float32 parameter gradients are 2.0e-4 from
  float64 where eager create_graph's are 8e-7. nequip's own
  `compile_mode="compile"` shows the same 2.0e-4: it is Inductor's float32
  rounding of the e3nn tensor products, not create_graph.
- Inductor's FX graph cache served stale backward kernels after a fake
  implementation's strides changed (the cache key does not include them).
- fairchem's `Safeacos` saves the clamped input it computes inside `forward`,
  which autograd treats as a constant: eager UMA training never differentiates
  its backward w.r.t. the input. The fused operator reproduces that.
- AllScAIP's blocks format their layer index into profiler labels, which a
  symbolic integer cannot do, so block scalars start as compile-time
  constants. UMA passes the number of atoms the same way: kept constant, every
  batch with a new atom count compiled its blocks again (2.3 s per step instead
  of 80 ms), hence a scalar that changes becomes dynamic.
- One compiled block function serves every checkpointed block: NequIP's
  convolutions differ in structure from layer to layer, which with the two
  grad modes and the duck-sizing recompile exceeded Dynamo's recompile limit
  of 8 (a hard error under `fullgraph`). Block functions get 8 per block.
- UMA specializes on the number of structures per batch, and MoLE (mixture of
  linear experts) splits edges per structure with data-dependent shapes: UMA
  runs with `num_moe_experts=0`.

## Benchmarks

`bench/tione_bench.sh` ran every (model, method) pair in its own process on one
dedicated 8-GPU TI-ONE node (RTX PRO 5000 72 GB, TF32 on as in training), one
lane per GPU, each lane pinned to the 32 CPU threads next to its GPU; no other
process shared a GPU. A step is a whole training step (forward, forces,
loss, backward) on a fresh batch of real OMol25 molecules; ms/step and peak
memory (allocator peak above parameters and inputs) are over 10 steps after 3
warm-up steps, which compile. Then the largest batch that fits (grow by 1.5x,
bisect to 5%) and the throughput there. Within a table every method sees the same batches.
Speedup = baseline ms / create_graph ms; memory saved = 1 - create_graph peak /
baseline peak (negative: create_graph uses more); max batch = ratio of the
largest batches. Raw results and tables: `bench/results/run-tione-20261018-0335/`.

Methods: `cg_eager` is eager create_graph through the same model code (what the
frameworks do); `cg`, `cg_budget` (memory budget 0.5) and `cg_ckpt`
(per-block checkpointing) are create_graph. Baselines: `explicit_*` is fairchem
`feature/explicit_create_graph` (the network's VJP written by hand, compiled;
plus budget 0.5; plus per-block checkpointing), `2f2b*` is fairchem
`feature/2f2b` (the dual forward written by hand, compiled; same knobs),
`native_eager` is metatrain's PET with its own autograd forces, and
`nequip_eager` / `nequip_compile` are nequip's force output in its eager and
`compile_mode="compile"` modes.

### AllScAIP md_NeNo (85M), 1024 atoms

| method | ms/step | peak GiB | max atoms | atoms/s at max |
|---|---|---|---|---|
| cg_eager | 1234.0 | 47.7 | 1536 | 800 |
| explicit_compiled | 745.6 | 57.5 | 1285 | 1433 |
| explicit_budget | 734.4 | 36.6 | 1928 | 1269 |
| explicit_ckpt | 1077.1 | 8.6 | 7975 | 720 |
| 2f2b | 915.2 | 45.0 | 1656 | 1133 |
| 2f2b_budget | 935.1 | 27.5 | 2485 | 1023 |
| 2f2b_ckpt | 995.0 | 20.6 | 3456 | 878 |
| **cg** | 658.6 | 47.3 | 1575 | 1584 |
| **cg_budget** | 710.6 | 40.8 | 1787 | 1459 |
| **cg_ckpt** | 1054.8 | 8.2 | 7391 | 794 |

The 2F2B lane measured create_graph on its own (the same batches): 666.3,
709.5 and 1068.7 ms, within 1.3% of the rows above.

| create_graph | vs | speedup | memory saved | max batch |
|---|---|---|---|---|
| cg | cg_eager | 1.87x | +1% | 1.03x |
| cg | explicit_compiled | 1.13x | +18% | 1.23x |
| cg | 2f2b | 1.37x | -5% | 0.95x |
| cg_budget | explicit_budget | 1.03x | -12% | 0.93x |
| cg_budget | 2f2b_budget | 1.32x | -49% | 0.72x |
| cg_ckpt | explicit_ckpt | 1.02x | +5% | 0.93x |
| cg_ckpt | 2f2b_ckpt | 0.93x | +60% | 2.14x |

Above 1024 atoms only the memory knobs fit (create_graph vs explicit):

| atoms | cg_budget | explicit_budget | cg_ckpt | explicit_ckpt |
|---|---|---|---|---|
| 1536 | 1146.9 ms, 62.6 GiB | 1121.7 ms, 56.1 GiB | 1645.1 ms, 12.3 GiB | 1692.4 ms, 13.0 GiB |
| 2048 | | | 2216.6 ms, 16.5 GiB | 2244.6 ms, 17.3 GiB |
| 3072 | | | 3396.3 ms, 25.0 GiB | 3469.0 ms, 26.2 GiB |

Memory budget sweep at 1024 atoms, create_graph / explicit: 0.3: 741.6 / 732.7
ms, 43.7 / 37.0 GiB; 0.5: 710.6 / 734.4 ms, 40.8 / 36.6 GiB; 0.7: 681.7 /
678.8 ms, 38.1 / 36.6 GiB.

Smaller AllScAIP models, 1024 atoms (ms/step, peak GiB):

| model | cg_eager | 2f2b | cg | cg vs cg_eager | cg vs 2f2b |
|---|---|---|---|---|---|
| h128_l2 | 191.4, 3.3 | 214.0, 2.4 | 186.4, 3.4 | 1.03x | 1.15x, -42% memory |
| h256_l4 | 279.6, 9.5 | 270.8, 8.4 | 266.5, 9.9 | 1.05x | 1.02x, -19% memory |
| h512_l6 | 508.9, 23.8 | 468.0, 22.1 | 387.3, 24.0 | 1.31x | 1.21x, -9% memory |

### PET (metatrain)

Default hypers at 4096 atoms; `large` (d_pet 256, d_node 512, 3 GNN layers)
at 2048 atoms (at 4096 only the memory knobs fit).

| PET | method | ms/step | peak GiB | max atoms | atoms/s at max | vs native_eager |
|---|---|---|---|---|---|---|
| default | native_eager | 518.4 | 33.8 | 8540 | 7015 | |
| | cg_eager | 498.5 | 33.8 | 8540 | 8001 | 1.04x |
| | **cg** | 335.4 | 31.3 | 10199 | 12425 | 1.55x, +7% memory, 1.19x batch |
| | **cg_budget** | 297.4 | 22.8 | 11576 | 11708 | 1.74x, +33%, 1.36x |
| | **cg_ckpt** | 483.3 | 20.0 | 12811 | 7440 | 1.07x, +41%, 1.50x |
| large | native_eager | 678.4 | 41.5 | 3762 | 2920 | |
| | cg_eager | 578.4 | 41.5 | 3762 | 3477 | 1.17x |
| | **cg** | 387.1 | 38.7 | 3858 | 5333 | 1.75x, +7%, 1.03x |
| | **cg_budget** | 356.5 | 27.4 | 5364 | 5573 | 1.90x, +34%, 1.43x |
| | **cg_ckpt** | 567.5 | 19.7 | 7271 | 3213 | 1.20x, +53%, 1.93x |

### NequIP (nequip, e3nn)

Presets S and M at 4096 atoms, L at 2048. On S every method reaches the search
limit (104976 atoms).

| NequIP | method | ms/step | peak GiB | max atoms | atoms/s at max | vs nequip_compile |
|---|---|---|---|---|---|---|
| S | nequip_eager | 30.7 | 1.9 | | 101546 | |
| | nequip_compile | 24.6 | 1.8 | | 357707 | |
| | cg_eager | 28.5 | 1.9 | | 153250 | |
| | **cg** | 15.0 | 1.8 | | 347518 | 1.64x, +3% memory |
| | **cg_budget** | 14.3 | 1.1 | | 347681 | 1.72x, +38% |
| | **cg_ckpt** | 21.5 | 1.4 | | 218206 | 1.14x, +25% |
| M | nequip_eager | 339.4 | 11.3 | 26716 | 11327 | |
| | nequip_compile | 149.8 | 10.9 | 27401 | 24751 | |
| | cg_eager | 349.6 | 11.3 | 26716 | 11298 | |
| | **cg** | 158.3 | 11.8 | 26047 | 19947 | 0.95x, -8%, 0.95x batch |
| | **cg_budget** | 146.1 | 10.4 | 28826 | 25758 | 1.03x, +4%, 1.05x |
| | **cg_ckpt** | 248.9 | 5.9 | 51633 | 13954 | 0.60x, +46%, 1.88x |
| L | nequip_eager | 946.3 | 20.9 | 7649 | 2005 | |
| | nequip_compile | 285.1 | 23.1 | 6570 | 6516 | |
| | cg_eager | 942.8 | 20.9 | 7649 | 2013 | |
| | **cg** | 318.0 | 23.0 | 6570 | 5923 | 0.90x, +0%, 1.00x |
| | **cg_budget** | 362.4 | 35.1 | 4725 | 5617 | 0.79x, -52%, 0.72x |
| | **cg_ckpt** | 603.0 | 8.5 | 18570 | 2955 | 0.47x, +63%, 2.83x |

### UMA (eSCN-MD, fairchem), 300 atoms

About 300 atoms per batch (the per-GPU batch of the `uma_s` training config),
with the same number of structures in every batch since UMA specializes on it;
no max-batch search (every size would compile again). These rows come from a
rerun on a one-GPU TI-ONE slot (`bench/results/run-tione-uma3-20261018-0533/`),
after the fix for block scalar arguments that change (in the dedicated run the
checkpointed blocks compiled again for every new atom count).

| UMA | method | ms/step | peak GiB | vs cg_eager |
|---|---|---|---|---|
| S (K4L2) | cg_eager | 90.4 | 5.3 | |
| | **cg** | 51.5 | 5.0 | 1.75x, +5% memory |
| | **cg_budget** | 50.5 | 4.9 | 1.79x, +8% |
| | **cg_ckpt** | 82.0 | 1.6 | 1.10x, +69% |
| M (K10L4) | cg_eager | 420.5 | 26.7 | |
| | **cg** | 240.1 | 27.0 | 1.75x, -1% |
| | **cg_budget** | 246.6 | 24.9 | 1.71x, +7% |
| | **cg_ckpt** | 445.9 | 3.8 | 0.94x, +86% |

### What the numbers say

**Against eager create_graph** (what the frameworks run), create_graph is
faster on every model at about the same memory: 1.87x on AllScAIP, 1.49x on
PET (1.55x and 1.75x against metatrain's own training forward), 1.9x to 3.0x
on NequIP, and 1.75x on UMA S and M.

**Against the hand-written explicit VJP** (AllScAIP), the generated graph is
1.13x faster with 18% less memory and a 23% larger largest batch. With
per-block checkpointing the two are equal (1.01x to 1.03x) at every batch size
up to 3072 atoms, create_graph 4-5% lower in memory. With a memory budget they
are equally fast, but create_graph keeps 4-18% more memory: the min-cut
partitioner saves more of the generated graph than of the hand-written one.

**Against hand-written 2F2B**, create_graph is 1.37x faster at 5% more memory,
and 1.32x faster under the same budget, where 2F2B goes much lower (27.5
against 40.8 GiB). With per-block checkpointing 2F2B is 7% faster, but
create_graph needs 60% less memory and fits 2.14x the batch.

**Against nequip's compile mode** create_graph is 1.64x faster on S, even on
M (0.95x; 1.03x with the budget) and 10% slower on L. nsys (three steps of
each, `bench/profile_pair.sh`, `bench/compare_kernels.py`) puts the gap on L in
40 extra small GEMM launches (+22 ms per step) and more Triton reduction time
(+17 ms); the other kernels match one to one. This fits a known cost of the
design: the region's differentiated inputs must require grad for its in-graph
`autograd.grad`, so AOTAutograd's backward of the region also computes the
loss gradient with respect to them (a second-order term through the force
graph), which nothing reads. nequip's compile mode traces the forces with
`make_fx` and compiles a graph whose position input does not require grad.
NequIP feeds the edge geometry into every layer, so that term repeats per
layer. AllScAIP's profile (`cg_budget` against `explicit_budget`, equal step
times) shows the same pattern at 2%: 12 ms more GEMM time per step in 40 extra
launches of the attention-block GEMM shapes, 4 per layer. A functorch-level
VJP (`torch.func.vjp`) inside the region would keep the inner differentiation
out of the outer backward; torch 2.13's Dynamo refuses the other way out,
leaves created inside the graph that outputs depend on.

**Memory knobs.** The budget lowers create_graph's peak on PET (by 27-29%) and
AllScAIP (14%) at little cost in time, sometimes a gain (PET, NequIP S and M
get faster: less memory traffic). On NequIP L it raises the peak (35 against
23 GiB), the partitioner's recompute materializing large tensor-product
intermediates in the backward; checkpointing is the knob there (8.5 GiB, 2.8x
the batch). Per-block checkpointing cuts AllScAIP from 47.3 to 8.2 GiB at 1.6x
the time.

## fairchem integration (`feature/create_graph`)

The worktree `mlip/fairchem-create_graph`, branch `feature/create_graph` from
`d9aad272` (the tip of `feature/2f2b`), commit `ad5eea427`:

- trainer: `conservative_force_mode: generated` next to `create_graph` and
  `2f2b` (`units/mlip_unit/conservative_generated.py`): `cg.create_graph(model,
  batch, outputs=<energy>, inputs=("pos",))`, forces `= -grads["pos"]`, then
  the usual `compute_loss` and `backward`. bf16 is refused, as for 2F2B;
- AllScAIP: in this mode the backbone network and the energy head are one
  `cg.compile` region and the head skips its own force computation; backbone
  switches `cg_compile`, `cg_memory_budget`, `cg_checkpointing` mirror the
  2F2B ones. UMA (eSCN-MD) the same, without MoLE and `otf_graph`;
- configs and launchers: `allscaip-scripts/omol_4m_8gpu_atoms1024_cons_generated`,
  `..._atoms1536_cons_generated` (budget 0.5), `..._atoms1024_cons_generated_ckpt`;
- `tests/core/models/allscaip/test_allscaip_generated.py`: float64 parameter
  gradients against create_graph (1e-10, with and without checkpointing,
  molecules and a periodic cell), batch keys untouched, stress refused, and the
  compiled region on GPU (default, checkpointing, budget) against create_graph
  for three batch shapes on the same graphs.

fairchem CLI training, AllScAIP md_NeNo, real OMol25, one GPU, 30 steps:
atoms/s over steps 10-29 from the trainer's log (optimizer, EMA and data
loading included) and nvidia-smi's peak (`bench/fairchem_e2e.sh`), on the
dedicated node and, second figure, on the development machine:

| config | atoms/s | peak GiB | generated vs it |
|---|---|---|---|
| `..._cons` (create_graph) | 978 / 1020 | 48.4 | 1.48x, same memory |
| `..._cons_2f2b` | 1169 / 1137 | 45.5 | 1.24x, 6% more memory |
| `..._cons_explicit` | 1482 / 1463 | 50.8 | 0.98x, 5% less memory |
| **`..._cons_generated`** | **1451 / 1461** | **48.3** | |
| `..._atoms1536_cons_2f2b` | 1065 / 853 | 46.3 | 1.30x, 39% more memory |
| `..._atoms1536_cons_explicit` (budget 0.5) | 1375 / 1374 | 51.9 | 1.01x, 24% more memory |
| **`..._atoms1536_cons_generated`** (budget 0.5) | **1388 / 1378** | **64.5** | |
| **`..._cons_generated_ckpt`** | **917 / 912** | **22.1** | |

Two GPUs, 4 steps (DDP), generated and create_graph: both ranks hold the same
gradient checksums (the trainer's `debug_checksums_save_path`) in both modes;
generated against create_graph, the loss agrees to 2e-5 and the per-parameter
gradient magnitudes to 1.2e-4 (norm over 409 parameters, worst 1.8e-3: TF32).
The 30 attention key-projection biases are left out: their gradient is zero in
exact arithmetic (softmax is invariant to a constant shift of a row) and both
modes hold rounding noise there. The CLI runs end with fairchem's
`ZeroDivisionError` in the validation metrics (`max_eval_steps_per_epoch=1`
leaves a metric without samples), after the measured steps.

## torch versions

`bench/version_matrix.sh` builds one venv per torch release (each with
metatrain 2026.4.1 and nequip 0.19.1, create_graph installed `--no-deps`) and
runs every mechanism test plus the real PET and NequIP tests on the smaller
sizes (accuracy in float32 and float64, memory knobs, the fused attention
operator, nequip's force output, one compile across batches; the epoch tests
and the largest sizes run on 2.13 only):

| torch | in-graph `autograd.grad` | result |
|---|---|---|
| 2.10.0 (cu128) | no: eager fallback | 47 passed, 16 compiled-only skipped: exact results, one warning |
| 2.11.0 (cu130) | yes | 63 passed |
| 2.12.1 (cu130) | yes | 63 passed |
| 2.13.0 (cu130) | yes | 63 passed |
| 2.14.1 (cu130) | yes | 63 passed |

Three torch bugs needed version-specific workarounds, all in
`compile_config.py`: on 2.11 Inductor's memory-coalescing analysis of tiling
asserts on dynamic ranges such as `(K+1)*N` (`triton.coalesce_tiling_analysis`
off); on 2.12 and 2.13 the mix-order reduction fails on them
(`triton.mix_order_reduction` off); on 2.11 and 2.12 building guards fails
when a symbol has no source location (NequIP's e3nn code), formatting a guard
comment (2.13's `SLoc.__str__`, backported inside regions).

## Layout

- `src/create_graph/`: `api.py` (`create_graph`, `compile`, `reset`),
  `region.py` (regions, input kinds, chaining, cache), `checkpoint.py`
  (per-block checkpointing), `ops.py` (`custom_op`), `compile_config.py` (the
  only place with private torch APIs), `pytree.py`, `testing.py`, `errors.py`
- `src/cg_bench/` (repository only, not in the wheel): real-model adapters
  (metatrain PET, nequip NequIP, fairchem AllScAIP and UMA), OMol25 data and
  packing, the training step and loss, the benchmark runner
- `kernels/edge_attn/`: the fused attention operator example
- `bench/`: `run_matrix.sh` (benchmark lanes), `summarize.py`, `fairchem_e2e.sh`
  and `summarize_e2e.py` (fairchem CLI end-to-end), `version_matrix.sh`,
  `ddp_parity.py`, `tione_bench.sh` / `lane.sh` (a whole run on one 8-GPU node),
  `tione_ddp.sh` (multi-node DDP on TI-ONE), results
- `tests/`: semantics, custom ops, torch surface, DDP, PET, NequIP, fairchem
