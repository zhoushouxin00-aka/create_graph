# create_graph

把能量 `E` 和坐标梯度 `dE/dx` 放进同一张前向图，用一次 `loss.backward()` 训练保守力模型。模型和 PyTorch 的求导公式保持原样，编译器可以一起优化能量与求力的计算。

常用接口只有两个：`cg.compile` 标记计算能量的网络，`cg.create_graph` 返回模型输出与输入梯度。

## 安装与最小用法

在自己的训练环境中安装，Python >= 3.10，依赖 `torch>=2.11`：

```bash
pip install "git+https://github.com/zhoushouxin00-aka/create_graph.git@online/for_initial_version"
```

下面是完整的训练步；`grads["pos"]` 是 `dE/dpos`，力需要取负号：

```python
import torch
from torch import nn
import create_graph as cg

network = cg.compile(nn.Sequential(nn.Linear(3, 16), nn.SiLU(), nn.Linear(16, 1)))

def model(batch):
    return {"energy": network(batch["pos"])}

batch = {"pos": torch.randn(8, 3)}
out, grads = cg.create_graph(model, batch)
force = -grads["pos"]
loss = out["energy"].square().mean() + force.square().mean()
loss.backward()
```

集成已有模型时，在初始化中写 `self.network = cg.compile(self.network)`，训练时调用 `cg.create_graph(model, batch)`。邻居表、边向量等几何计算可以留在网络前面，继续使用原来的 autograd。包含区域外几何计算的完整示例见 [`examples/mlp.py`](examples/mlp.py)。

调用约定：

- 模型只计算能量，关闭自身的求力头。一次调用只支持一个依赖所选输入的 `cg.compile` 区域。
- 默认对输出的 `energy`、输入的 `pos` 求导；输入支持字典或属性对象。输出本身是张量时传 `outputs=None`；嵌套输出可传 `outputs=lambda out: out["energy"]["energy"]`。
- 编译区域应返回最终用于求导的能量。能量缩放或额外依赖坐标的能量项应放在区域内。
- DDP 训练时将 DDP 包装后的模型传给 `cg.create_graph`。

验证或分子动力学只需要力、不需要参数梯度时，传 `train=False`，返回的输出和梯度都会脱离计算图：

```python
out, grads = cg.create_graph(model, batch, train=False)
```

## 它怎样工作

通常的保守力训练先用 `torch.autograd.grad(..., create_graph=True)` 求力，再对能量和力组成的 loss 求参数梯度。`cg.compile` 标记的区域把网络前向与求力合成一张普通的一阶图，后续反向由 AOTAutograd 生成、Inductor 编译，激活重算也可以作用于求力部分。

```mermaid
flowchart LR
  geom["邻居和几何，eager"] --> region["cg.compile：E 与 dE/dx"]
  region --> loss["loss(E, F)"]
  loss --> paramGrad["loss.backward()：参数梯度"]
```

## 示例与检查

在已安装本包、pytest 的环境中：

```bash
python examples/mlp.py
python -m pytest tests -m "not gpu"
```

测试覆盖能量、坐标梯度、参数梯度、动态形状、checkpoint、自定义算子和两进程 CPU DDP。`cg.create_graph(..., exact=True)` 可用于与 eager 求导对照。

普通算子的二阶导由 PyTorch 完成。需要支持二次求导的融合 kernel 可用 `cg.custom_op` 登记 `forward1`、`backward1`、`backward2`；见 [`kernels/edge_attn/ops.py`](kernels/edge_attn/ops.py)。pip 安装包只包含 `src/create_graph/`，不包含该 kernel 示例、训练数据和 benchmark。
