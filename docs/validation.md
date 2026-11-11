# AllScAIP 长训记录

核对日期：2026-11-11，北京时间。本文区分完成训练、日志覆盖范围与尚未完成的验证。

## 配置与四次运行

- 硬件：2 节点 × 8 张 NVIDIA RTX PRO 5000 72GB Blackwell。
- 模型：AllScAIP，hidden size 512、6 层、8 heads，保守力训练，attention 配置为 `memory_efficient`。
- 数据：OMol25 train_4M；每 rank 的 `max_atoms=1024`；计划训练 12 epoch，每 epoch 13995 step，共 167940 step。
- TF32 / FP32 按训练配置命名。每轮验证限制为 1 step（`max_eval_steps_per_epoch=1`）。

| 模式 | 开始时间 | 已核实的训练范围 | 结束证据 | TI-ONE 任务 ID |
|---|---|---|---|---|
| eager TF32 | 10-31 23:29:50 | 12 epoch / 167940 step | 11-03 12:08:09，exit=0 | `train-1707555358162411392` |
| create_graph TF32 | 11-06 21:09:08 | 12 epoch / 167940 step | 11-08 16:00:09，exit=0 | `train-1709722672231623424` |
| eager FP32 | 11-04 04:38:32 | 日志至 147500 step，完成 10 epoch | 已收集的平台状态为 STOPPED | `train-1709467592555244288` |
| create_graph FP32 | 11-08 16:00:37 | 日志至 165100 step，完成 11 epoch | 本地未见结束状态文件 | `train-1709722703353359104` |

最后一条训练日志的步数与完成的总步数不同：TF32 日志每 100 step 采样，最后采样为 167900，最终 checkpoint 和正常退出确认总计 167940 step。FP32 两组不计作完成 12 epoch。

## 速度口径

| 精度 | 对齐范围 | eager | create_graph | 加速比 |
|---|---|---|---|---|
| TF32 | 完整 12 epoch，启动至正常退出 | 60.6386 h | 42.8503 h | 1.415× |
| FP32 | 前 10 epoch，`step_0` 至 `step_139950` 的 checkpoint `.metadata` 修改时间差 | 61.1730 h | 45.3894 h | 1.348× |

加速比为 eager 耗时除以 create_graph 耗时。TF32 包含启动、编译、训练和 checkpoint 等开销；FP32 使用双方相同训练进度的 checkpoint 时间差，不能与 TF32 的完整启动到退出口径混为一谈。这里没有将未完成训练的总耗时外推为 12 epoch，也没有混入后来 `max_atoms=768` 的扩展测试。

## 遇到的问题与修复

旧生成版在 4 卡排查运行 `ddp-loss-4gpu-generated-20261102` 的 step 1200 附近出现 OOM：申请 44 MiB 失败时，PyTorch 已分配 69.71 GiB，GPU 总容量为 71.12 GiB。证据在 `log/node0.log` 第 220、276 行。早期 16 卡 compiled 运行还出现 NCCL collective timeout；该日志本身不足以认定超时的根因就是 OOM。

11 月 3 日的修复集中在联合图保存的激活和融合工作区：

- 联合图默认激活预算降至 `0.3`，保存集超过 24 GiB 时增加重算，优先保留矩阵乘和自定义算子。
- 对动态原子数、邻居数提前标记动态维度，避免为每种长度持续生成新图。
- 限制跨层融合距离、融合节点数和读取 buffer 数；阻止大读取与 reduction 横向融合，以尽早释放工作区。
- 默认禁用 CUDA graph 私有池，并增加显存和图数量采样日志。

两次后续 create_graph 长训均打印了 `partition request=0.3` 和 `fusion_distance=8 max_fusion_size=8 fusion_read_cap=8`。TF32 有 1680 次显存采样、FP32 有 1652 次；编译后采样图数量保持为 1。两组记录的窗口峰值最大为 58.01 GiB，reserved 最大为 61.01 GiB；这些是采样观测，不能视为任意输入的显存上限。

## 数值与模型质量边界

两节点日志中，先对同一步的 16 rank loss 取平均，再计算 `abs(eager - create_graph) / abs(eager)`：TF32 的 1680 个配对记录点相对差中位数约 5.36%，FP32 的 1476 个重叠记录点约 5.34%。FP32 在 step 700 已出现明显 grad norm 分叉。训练设置为 `deterministic=false`；这些观测没有证明分叉的根因。

因此，长训结果验证了该配置下的速度和持续运行表现，**没有证明长程训练轨迹逐步等价，也没有完成最终模型质量等价验证**。每轮 1 step 的验证不能替代完整验证集评估。

## 版本与证据位置

本次发布保留 `ad3c15a859d5c8a70e6ed3daca14d821256f513b` 的训练实现，仅整理两处 import 格式、文档及本地生成文件的忽略规则。需要固定该实现时可安装：

```bash
pip install "git+https://github.com/zhoushouxin00-aka/create_graph.git@ad3c15a859d5c8a70e6ed3daca14d821256f513b"
```

两条生成版启动脚本通过 `PYTHONPATH` 加载共享的 `create_graph/src`，日志中的预算和融合参数与该 commit 一致。但当时没有保存 create_graph SHA 或独立源码快照，不能逐字节认证历史进程加载的版本。

原始记录保留在本地相邻的 `fairchem-create_graph/logs/`，没有随 Python 包分发：

| 模式 | 日志目录 |
|---|---|
| eager TF32 | `ddp-loss-16gpu-eager-20261031/` |
| create_graph TF32 | `ddp-loss-16gpu-generated-tf32-20261103/` |
| eager FP32 | `ddp-loss-16gpu-eager-fp32-20261103b/` |
| create_graph FP32 | `ddp-loss-16gpu-generated-fp32-20261103/` |

各目录的 `node.txt` 记录硬件与启动时间，`log/node0.log`、`log/node1.log` 记录训练与显存，`work/runs/canonical_config.yaml` 记录配置，`work/runs/checkpoints/` 提供进度证据。TF32 目录另有正常退出的 `status.txt`。启动脚本位于相邻仓库的 `allscaip-kernels/bench/ddp_loss_16gpu_one.sh`、`ddp_loss_16gpu_generated_fp32.sh`。

## 本次发布检查

Python 3.12.14、PyTorch 2.14.1+cu130 下，49 项非 GPU 测试通过，覆盖求导语义、动态形状、checkpoint、自定义算子和两进程 CPU DDP。DDP 在允许本地 socket 通信的环境中运行。wheel 已构建并独立安装，验证了小模型前向、求力和参数反传；这些检查不替代新的 GPU 长训。
