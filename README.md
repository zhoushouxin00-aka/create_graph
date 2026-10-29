# create_graph

保守力训练要先对坐标求导，再对参数求导。`create_graph` 把前一步写进前向图，后面就和普通训练一样：一次 `loss.backward()`。

装进你自己的训练环境，不要把这个仓库拷进你的项目：

```bash
pip install git+https://github.com/zhoushouxin00-aka/create_graph.git
```

只依赖 `torch>=2.11`。torch 2.10 及更早会退回原来的 eager 求力，结果仍对，只是不编译。

## 它在做什么

保守力是两步求导。模型算出能量 `E`，力是 `F = -dE/dx`。loss 同时看能量和力，再对参数求导，所以参数梯度要穿过“对坐标的那一次求导”。

手写的做法是：

```python
energy = model(batch)["energy"]
force = -torch.autograd.grad(energy.sum(), batch["pos"], create_graph=True)[0]
loss = energy_loss(energy) + force_loss(force)
loss.backward()
```

求力本身已经是一张反向图。后面的编译器看不到它，显存预算和逐 block 重算也作用不到它上面。

`create_graph` 不另写一套求导规则。算能量的那段用 `cg.compile` 标出来。调用 `cg.create_graph` 时，这段的前向和 `dE/dx` 合成一张普通的一阶图。`loss.backward()` 就是对这张图的普通反向。邻居表、边向量这些几何留在区域前面，仍走原来的 autograd。

```mermaid
flowchart LR
  geom["邻居和几何，eager"] --> region["cg.compile 的网络：E 和 dE/dx 在同一张图里"]
  region --> loss["loss(E, F)"]
  loss --> paramGrad["loss.backward()：对参数的普通反向"]
```

AllScAIP 85M、单卡、一步训练：编译后大约是 eager create_graph 的 1.87 倍。放进 fairchem 的训练循环后大约 1.48 倍。峰值显存和 eager 差不多；要省显存，用下面的 memory budget 或 checkpoint。多卡加速还没有测。

## 怎么用

模型在这次调用里只算能量，把自己的力头关掉。`cg.compile` 包住最终用来求导的那个能量。默认对输出里的 `energy` 求导，对输入里的 `pos` 求导。

```python
import create_graph as cg

self.network = cg.compile(self.network)

out, grads = cg.create_graph(model, batch)
loss = energy_loss(out["energy"]) + force_loss(-grads["pos"])
loss.backward()
```

[`examples/mlp.py`](examples/mlp.py) 是上面这三行的完整脚本：一个小网络，坐标先在区域外做一次中心化，再进编译过的网络。

```bash
python examples/mlp.py
```

输出如果是能量张量本身，而不是带 `energy` 键的字典，调用时写 `outputs=None`。fairchem 那种 `{"energy": {"energy": E}}`，传一个把张量取出来的函数：`outputs=lambda out: out["energy"]["energy"]`。

## 仓库里有什么

pip 只会装走 `src/create_graph/`。

| 路径 | 作用 |
|---|---|
| `src/create_graph/api.py` | `create_graph` 和 `compile`，集成时用这两个 |
| `src/create_graph/region.py` | 把区域的前向和 `dE/dx` 放进同一张图 |
| `src/create_graph/compile_config.py` | 编译选项。整个包里只有这个文件碰 torch 的私有接口 |
| `src/create_graph/ops.py` | 融合算子登记。普通模型用不到 |
| `examples/mlp.py` | 最小可运行用法 |
| `tests/test_semantics.py`、`test_custom_ops.py`、`test_ddp.py`、`test_torch_surface.py` | 不装大模型也能跑的测试 |
| `kernels/edge_attn/` | 融合 attention 的登记例子，不进安装包 |

```bash
python -m pytest tests -m "not gpu"
```

## 进阶

显存不够时，给区域一个预算，或者按子模块路径做逐 block 重算：

```python
self.network = cg.compile(
    self.network,
    config=cg.CompileConfig(memory_budget=0.5, checkpoint=r"blocks\.\d+"),
)
```

`memory_budget` 是 `(0, 1]` 里的激活显存预算。`checkpoint` 是对子模块路径的正则（`re.fullmatch`）：匹配到的 block 只留输入，反传时重算。默认 `backend="inductor"`、`dynamic=True`，一般不用改。

验证或分子动力学只要力、不要参数梯度时：

```python
out, grads = cg.create_graph(model, batch, train=False)
```

这时力和输出都不带计算图。

普通算子的二阶导由 PyTorch 自己完成。只有一个融合 kernel 看不进计算图、又必须求两次导时，才用 `cg.custom_op` 登记 `forward1`、`backward1` 和 `backward2`。[`kernels/edge_attn/ops.py`](kernels/edge_attn/ops.py) 是带 mask 的 attention 例子。
