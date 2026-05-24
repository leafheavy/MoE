# Heterogeneous Two-Stage MoE (Mixture-of-Experts)

该仓库实现了一个**异构两阶段路由的 MoE 语言模型**，核心思想是：

- **Stage-1（池级路由）**：先判断 token 更适合“稳定专家池（stable）”还是“迁移专家池（transfer）”。
- **Stage-2（池内路由）**：在对应专家池中进行 noisy top-k 专家选择与加权融合。

相比单阶段 MoE，该设计把“任务难度/类型分流”和“专家细粒度选择”解耦，便于做可解释性分析与消融研究。

---

## 1. 代码结构

```text
.
├── model.py        # 模型定义：Expert / TwoStageRouter / HeterogeneousMoEModel
├── train.py        # 训练、评估、日志、checkpoint、ablation 入口
├── dataset.py      # synthetic + HuggingFace 数据集包装与 DataLoader
├── utils.py        # 辅助损失、统计指标、学习率调度、随机种子
├── test_model.py   # 单元测试与基础集成测试
└── config.yaml     # 默认实验配置
```

---

## 2. 模型机制与关键公式

### 2.1 两阶段路由

给定 token 表示 `h`：

1. **Stage-1（二分类）**
   - 输出 `p(pool|h) ∈ R^2`，并通过 hard argmax 把 token 分配到 stable / transfer 池。
2. **Stage-2（池内 top-k）**
   - 在选定池内对专家 logits 做 softmax + top-k 截断。
   - 对 top-k 权重重归一化，再对专家输出加权求和。

### 2.2 异构专家池

- stable experts：较小 FFN（`d_ff_stable`），偏向高频/简单模式。
- transfer experts：较大 FFN（`d_ff_transfer`），偏向复杂模式与迁移能力。

### 2.3 训练目标

总损失由 LM 主损失和辅助损失组成：

- `L_total = L_lm + λ_ent * L_entropy + λ_bal * L_balance`

其中：

- `L_entropy`：鼓励路由更“确定”（低熵）。
- `L_balance`：鼓励专家负载均衡（Switch 风格）。

---

## 3. 快速开始

## 3.1 环境准备

建议 Python 3.10+。

```bash
pip install torch pyyaml tensorboard
# 若使用 HuggingFace 数据：
pip install datasets transformers
```

## 3.2 训练

默认配置文件是仓库根目录的 `config.yaml`：

```bash
python train.py --config config.yaml
```

可选参数：

- `--resume <ckpt_path>`：断点恢复
- `--run_ablations`：训练后跑内置消融
- `--no_entropy_loss`：关闭 entropy 辅助项

## 3.3 运行测试

```bash
python -m pytest -q
```

---

## 4. 配置说明（`config.yaml`）

配置按模块分组：

- `model`：基础 Transformer 参数（`d_model`, `n_layers`, `max_seq_len` 等）
- `moe`：专家池规模与路由参数（`n_stable`, `n_transfer`, `top_k`, `noise_std`）
- `training`：优化器与训练过程（`lr`, `max_steps`, `warmup_steps`, `lambda_*`）
- `data`：数据集来源（`synthetic | wikitext | c4`）
- `logging`：TensorBoard / W&B 相关参数

> 若使用 `wikitext/c4`，请设置 `data.tokenizer_name`（例如 `gpt2`）。

---

## 5. 关键日志与分析指标

训练和验证阶段会记录以下指标，建议研究时重点跟踪：

- `train/loss_lm`, `train/loss_entropy`, `train/loss_balance`, `train/loss_total`
- `val/val_ppl`, `val/val_loss`
- 路由统计：
  - `stable_frac`, `transfer_frac`（池分流比例）
  - `stable_entropy`, `transfer_entropy`（池内路由熵）
  - `stable_load_std`, `transfer_load_std`（专家负载不均衡程度）

可用 TensorBoard 查看：

```bash
tensorboard --logdir logs/
```

---

## 6. 面向研究者的实验建议

### 6.1 推荐的消融矩阵

1. **路由机制消融**
   - 两阶段 vs 单阶段（标准 top-k MoE）
   - hard Stage-1 vs soft Stage-1（可作为后续改造）
2. **异构性消融**
   - `d_ff_stable == d_ff_transfer`（退化成同构 MoE）
   - 调节 `n_stable:n_transfer` 比例
3. **正则项消融**
   - `λ_ent ∈ {0, 1e-3, 1e-2, 1e-1}`
   - `λ_bal ∈ {0, 1e-3, 1e-2, 1e-1}`
4. **路由稀疏度消融**
   - `top_k ∈ {1, 2, 4}`

### 6.2 建议补充的评估维度

- 参数效率：PPL / 参数量
- 计算效率：tokens/s、FLOPs（可集成 `fvcore` 或 `ptflops`）
- 路由稳定性：不同 seed 下专家占用方差
- 泛化迁移：跨域验证集 PPL

### 6.3 可继续扩展的方向

- 替换 Stage-1 为 Gumbel-Softmax 或可学习阈值门控
- 引入 capacity factor 与 token dropping 策略
- 增加 router z-loss / auxiliary clipping 稳定训练
- 实现 expert-level gradient norm 监控与动态重加权

---

## 7. 已修复的问题（相对原实现）

1. `train.py` 默认配置路径与仓库实际结构不一致（原默认 `config/config.yaml`）。
2. 使用 HuggingFace 数据时，训练流程未自动构建 tokenizer，导致断言失败。
3. `test_model.py` 仍使用 `src/` 目录假设，和当前仓库扁平结构不一致。

这些问题已在当前版本中修复，便于直接在仓库根目录运行。

---

## 8. 复现实验建议

- 固定 `seed` 并记录硬件环境（GPU 型号、CUDA、PyTorch 版本）。
- 对每个配置至少跑 3 个 seed，报告均值±方差。
- 同时保存：
  - checkpoint
  - config 快照
  - TensorBoard 日志
  - ablation summary（json）

---

## 9. 引用与致谢

若你的研究基于本仓库，请在论文/报告中说明：

- 使用了「Heterogeneous Two-Stage MoE」实现；
- 给出你使用的 commit hash 与配置文件。

