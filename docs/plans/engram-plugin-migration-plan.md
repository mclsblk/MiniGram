# MiniGram Engram 插件化与多通道迁移计划

## 1. 目标与边界

本次迁移将 `model/model_minigram.py` 中的 Engram、残差通道和模型组装逻辑拆开，使 MiniGram 能用一套统一接口展示并运行三类配置：

- 当前 MiniGram 的简化 Engram，作为行为和旧权重兼容基线；
- Qwen3.8-Flash 风格的记忆 gram 与 GR4 通道；
- DeepSeek-V4.1-Flash 风格的 Engram 与 mHC4 通道。

迁移后的代码应满足四个目标：

1. 主模型文件只保留模型组件定义、层级连接和最终组装。
2. Engram 内部步骤可独立替换，卷积、token compression 等组件可以按需启用或移除。
3. single、GR4 和 mHC4 通过相同的通道接口接入，不把条件分支散落到 Transformer 层的 `forward` 中。
4. 默认配置完整保持当前 MiniGram 的数值行为、调用方式和旧 `.pth` 权重兼容性。

本阶段只迁移模型侧结构。训练器继续使用项目现有逻辑，不引入 Muon 等新优化器、GRPO 或训练时冻结 Engram 的策略。

## 2. 目标文件结构

采用接近 DeepSeek 原始实现的精简结构，只新增两个模型文件：

```text
model/
├── model_minigram.py   # 基础模块、Transformer 层、MiniGram 组装
├── engram.py           # Engram 组件、预设、状态与构建器
└── channels.py         # single、GR4、mHC4 通道实现
```

不继续拆出 `hash.py`、`memory.py`、`kernel.py` 或单独的配置目录。Engram 的紧密相关实现集中在 `engram.py`，通道逻辑集中在 `channels.py`，以便学习和对照论文。

## 3. 总体组装方式

模型配置使用两个正交选择项：

```python
MiniGramConfig(
    use_engrams=True,
    engram_variant="qwen",
    engram_overrides={
        "postprocessor": "identity",
    },
    residual_variant="gr4",
)
```

- `engram_variant` 选择记忆 gram 的默认拓扑和算法。
- `engram_overrides` 覆盖预设中的单个组件，用于消融和教学展示。
- `residual_variant` 选择残差通道系统。
- `engram_n_layer_list` 独立决定在哪些 Transformer 层插入 Engram。

预设决定算法结构，本地实验规模由 `engram_vocab_size`、bucket 数、head 数和 head dimension 等参数控制。复现核心机制时不创建论文规模的巨大查找表。

首个完整版本提供以下预设：

| Engram 预设 | n-gram | token 映射 | 哈希 | 读出 | 后处理 | 默认插入位置 |
| --- | --- | --- | --- | --- | --- | --- |
| `legacy` | 保持现状 | 原始 token | 当前实现 | 当前门控与归一化 | 当前卷积行为 | attention 后 |
| `qwen` | 2/3-gram | 原始 token | prime rolling/XOR | signed-sqrt gate | dilated causal convolution | attention 前 |
| `deepseek` | 2/3/4-gram | token compression | prime rolling/XOR | signed-sqrt gate | identity | attention 前 |

通道预设为：

| 通道 | 通道数 | 核心行为 | 第一阶段目标 |
| --- | ---: | --- | --- |
| `single` | 1 | 当前标准残差流 | 与现有实现逐元素兼容 |
| `gr4` | 4 | 低秩逐维读取、分支写入、最终混合 | 复现 Qwen 的核心通道拓扑 |
| `mhc4` | 4 | pre/post/comb 映射、Sinkhorn 双随机混合、跨层 carry | 复现 DeepSeek 的核心通道拓扑 |

推荐的演示组合是 `legacy + single`、`qwen + gr4` 和 `deepseek + mhc4`，但配置层不硬编码这三个配对。用户可以组合其他 Engram 与通道，用于消融实验。

## 4. Engram 插件结构

### 4.1 `EngramSpec`

`engram.py` 定义一个轻量配置对象 `EngramSpec`。它描述五个可替换插槽，并包含 n-gram 阶数、边界规则、门控参数、卷积参数和插入位置等必要设置。

五个插槽为：

```text
token_mapper  -> hasher -> memory_store -> readout -> postprocessor
```

建议的内置组件：

| 插槽 | 第一阶段实现 |
| --- | --- |
| `token_mapper` | `identity`、`compressed` |
| `hasher` | `legacy`、`rolling_xor` |
| `memory_store` | `separate`、`packed` |
| `readout` | `legacy`、`signed_sqrt_rms`、`signed_sqrt_weighted` |
| `postprocessor` | `identity`、`causal_conv` |

预设只负责生成完整 `EngramSpec`。`engram_overrides` 在构造模块之前修改 spec，因而 `forward` 不需要根据字符串选择算法。

示例：

```python
# Qwen 核心，但去掉卷积
MiniGramConfig(
    engram_variant="qwen",
    engram_overrides={"postprocessor": "identity"},
)

# DeepSeek 核心，但保留原始 token id
MiniGramConfig(
    engram_variant="deepseek",
    engram_overrides={"token_mapper": "identity"},
)
```

### 4.2 统一数据流

Engram 的内部数据流固定为：

```text
input_ids
   │
   ▼
TokenMapper
   │ mapped_ids
   ▼
NGramHasher
   │ bucket_ids
   ▼
MemoryStore
   │ retrieved vectors
   ▼
Readout
   │ gated memory
   ▼
PostProcessor
   │
   ▼
delta: [batch, sequence, residual_channels, hidden_size]
```

统一输出为对残差流的增量 `delta`。single 通道时 `residual_channels=1`，GR4 和 mHC4 时通常为 4。Transformer 层只消费这个结果，不理解 token compression、哈希、查表或卷积细节。

### 4.3 token compression

DeepSeek 风格的 `CompressedTokenMapper` 使用持久化的 `token_map` buffer：

- 提供离线构造映射表的 helper；
- 提供模型级 setter，允许训练脚本注入由语料统计得到的映射；
- 将映射表保存进 `state_dict`，确保推理时无需重新统计语料；
- 未配置映射时显式报错或使用配置声明的 fallback，避免静默改变语义。

token compression 只改变哈希前的 token id，不修改 tokenizer、模型输入或语言模型词表。

### 4.4 哈希与边界

`rolling_xor` 负责多阶 n-gram bucket 地址计算，并将以下规则集中到同一个模块：

- 各阶 n-gram 使用独立的 prime 或 seed；
- 使用稳定的整数运算，避免 Python 进程级 hash 随机性；
- BOS、padding、序列开头和增量解码共享同一套边界定义；
- 全序列 forward 与逐 token decode 得到相同 bucket id。

旧算法保留在 `legacy` hasher 中，以保证迁移前后行为一致。

### 4.5 memory store

`MemoryStore` 只负责根据 bucket id 读取向量：

- `separate` 为不同 n-gram 阶数保留独立表，适合保持旧结构和教学展示；
- `packed` 将多个表组织为统一参数或统一索引空间，便于接近论文实现并减少 Python 调度。

两者对上层暴露相同的 shape 和语义。容量参数保持本地可运行，不默认复制论文中的完整参数规模。

### 4.6 readout 与门控

读出组件负责把检索向量与当前 hidden state 结合：

- `legacy` 原样迁移当前 MiniGram 的投影、门控和归一化；
- `signed_sqrt_rms` 实现 Qwen/DeepSeek 共有的 signed-sqrt gate 与 RMS 类归一化；
- `signed_sqrt_weighted` 在统一门控基础上提供多阶结果的可学习加权。

组件输出 shape 必须一致，使卷积和通道层无需知道具体门控类型。

### 4.7 后处理

`PostProcessor` 提供两个首发实现：

- `identity`：直接返回读出结果，对应不使用局部卷积的 DeepSeek 风格配置；
- `causal_conv`：实现 Qwen 风格的多尺度膨胀因果卷积，并维护增量推理状态。

移除卷积时不构造卷积参数，也不创建无用 cache。Qwen 的最大历史窗口由实际 kernel size 和 dilation 推导；若首版参数与论文配置一致，预期需要保留 9 个历史位置，但代码不应写死该数字。

### 4.8 插入位置

Engram 插入位置是预设的一部分：

- `legacy` 默认保持当前 MiniGram 的 attention 后插入；
- `qwen` 和 `deepseek` 默认在 attention 前注入。

Transformer 层可以保留一个清晰的结构分支来选择插入阶段，但不得在该分支内实现具体 Engram 算法。构建时将插入阶段解析为枚举或固定 callable，避免每个 token 重复解析字符串。

## 5. 通道插件结构

### 5.1 统一表示

`channels.py` 定义统一的通道载体：

```python
@dataclass
class ChannelState:
    streams: torch.Tensor       # [B, S, R, D]
    pre_mix: torch.Tensor | None = None
```

其中：

- `B`：batch size；
- `S`：sequence length；
- `R`：残差通道数；
- `D`：hidden size。

attention 和 FFN 仍只接收 `[B, S, D]`。通道插件负责从多个 residual streams 读取单一 hidden state，并把子层输出写回多个 residual streams。

Engram 统一生成 `[B, S, R, D]` 的增量，因此它可以接入三种通道而无需了解通道混合算法。

### 5.2 统一生命周期

每个通道实现提供等价的构建期和运行期接口：

```text
initialize(hidden) -> ChannelState
read(state, branch) -> hidden
write(state, branch_output, branch) -> ChannelState
inject(state, engram_delta) -> ChannelState
finalize(state) -> hidden
```

`branch` 用于区分 attention、FFN 和必要的 Engram 插入点。具体类可以将 read/write 融合实现，但 Transformer 层看到的调用语义保持一致。

### 5.3 single

`SingleResidualChannel` 是兼容基线：

- `R=1`；
- 初始化、读取、写回和最终输出应退化为当前标准 residual add；
- 不引入额外可训练参数；
- 默认配置下必须与迁移前输出逐元素一致。

### 5.4 GR4

`GR4Channel` 首版实现 Qwen 报告中与模型拓扑直接相关的核心机制：

- 四条 residual streams；
- 每个子层前通过低秩、逐维的可学习映射读出一个 hidden state；
- attention、FFN 和 Engram 输出通过各自的写入映射回到四条流；
- 模型末端使用最终 mixer 合并四条流。

GR4 的参数初始化应接近 single residual 的稳定行为，使小模型在接入多通道后可以正常开始训练。所有矩阵和低秩参数属于 channel module，不写入 attention 或 FFN 类。

### 5.5 mHC4

`MHC4Channel` 首版实现 DeepSeek mHC 的核心结构：

- 四条 residual streams；
- pre、post 和 composition 映射；
- 使用纯 PyTorch Sinkhorn 迭代把混合矩阵投影到近似双随机矩阵；
- 维护跨层使用的 carry 或 pre-mix 状态；
- 最终归并为单一 hidden state。

首版不创建额外 CUDA kernel 文件。纯 PyTorch 版本优先保证公式、梯度和结构清晰，性能优化留给后续独立阶段。

## 6. 配置与构建器

`MiniGramConfig` 新增或整理以下字段：

```python
use_engrams: bool
engram_variant: Literal["legacy", "qwen", "deepseek"]
engram_overrides: dict[str, Any]
engram_n_layer_list: list[int]
residual_variant: Literal["single", "gr4", "mhc4"]
residual_channels: int
```

现有 Engram 容量、维度和卷积字段继续保留，逐步映射到 `EngramSpec`。配置解析遵循以下顺序：

1. 读取 `engram_variant` 的完整预设；
2. 应用当前 `MiniGramConfig` 中显式设置的容量参数；
3. 应用 `engram_overrides`；
4. 验证组合是否合法；
5. 一次性构造最终组件。

验证器至少检查：

- n-gram 阶数非空且递增；
- memory store 数量与 n-gram 阶数匹配；
- 卷积只接收它支持的输入布局；
- `single` 的通道数固定为 1；
- `gr4` 和 `mhc4` 首版固定为 4；
- hidden size 可以被需要的 head 或 group 设置整除；
- compressed mapper 已获得合法 token map。

`model_minigram.py` 只调用类似以下构建器：

```python
self.channel = build_residual_channel(config)
self.engrams = build_engram_layers(config)
```

具体组件选择不散落到 block 的 `forward` 中。

## 7. 推理状态与 cache

Engram 使用显式状态对象：

```python
@dataclass
class EngramState:
    hash_tail: torch.Tensor | None
    post_state: Any | None

    def reorder(self, beam_idx: torch.Tensor) -> "EngramState": ...
```

- `hash_tail` 保存构造跨 decode step n-gram 所需的最近 token；
- `post_state` 保存因果卷积等后处理组件的历史；
- `reorder()` 统一处理 beam search 的 batch 重排。

模型 cache 新接口使用统一的 `engram` 字段。迁移期可以读取旧运行时 cache key，但新输出只生成新格式，避免长期维护两套状态协议。

全序列训练、prefill 和逐 token decode 必须共享同一套组件实现。不得为 decode 复制一份独立哈希或卷积公式。

## 8. 旧权重兼容策略

默认的 `legacy + single` 组合承担兼容责任。

采用 `load_state_dict` pre-hook 或等价的集中映射函数，将旧参数名转换到新模块路径。映射范围包括：

- Engram embedding 表到 `memory_store`；
- K/V 或门控投影到 `readout`；
- Engram norm 与 gate bias；
- 旧卷积参数到 `postprocessor`；
- 模型级 `norm_engram` 等受拆分影响的名称。

兼容层只负责旧名称迁移，不猜测 qwen 或 deepseek 预设。旧 checkpoint 未携带新配置字段时，明确使用 `legacy + single`。

迁移完成后应保留原有公共 import 路径。若项目外部代码从 `model_minigram.py` 导入旧 Engram 类，则在主文件中提供重导出别名，并在注释中标明兼容用途。

## 9. 分阶段迁移步骤

### 阶段 0：建立可比较基线

在可用的 PyTorch 环境中，用固定 seed、tiny config、`dropout=0` 记录：

- 当前 `state_dict` key 与 shape；
- 固定 input ids 的 logits；
- Engram 中间输出；
- prefill 后逐 token decode 的输出和 cache 结构；
- 有无 Engram 时的模型参数量。

这些结果作为重构期间的兼容基线。当前 shell 若没有 PyTorch，应先记录环境限制，并在项目已有训练环境中完成该步骤，不能用静态检查代替数值基线。

### 阶段 1：原样抽取 legacy Engram

1. 新建 `model/engram.py`。
2. 将现有 Engram 相关类和 helper 原样移动进去。
3. 在 `model_minigram.py` 保留兼容导入。
4. 不改变公式、参数名映射、调用顺序和 cache 语义。
5. 对照阶段 0 验证 logits 与中间输出。

这一阶段只改变代码位置，为后续插件边界建立可信基线。

### 阶段 2：引入统一 Engram 管线

1. 定义 `EngramSpec`、五类组件协议和 builder。
2. 用 adapter 把旧实现包装为 `legacy` 预设。
3. 将 Engram 输出统一成 `[B, S, R, D]`。
4. 引入 `EngramState`，同时保留旧 cache 的读取兼容。
5. 再次验证 `legacy + single` 的数值等价性。

### 阶段 3：引入通道 seam

1. 新建 `model/channels.py`。
2. 定义 `ChannelState` 和统一生命周期。
3. 实现 `SingleResidualChannel`。
4. 把 Transformer 层中的 residual add 改为通道 API 调用。
5. 验证默认模型仍逐元素等价。

这是风险最高的结构接缝，应在实现 GR4/mHC4 之前单独完成和验证。

### 阶段 4：实现 Qwen + GR4

1. 实现 identity token mapper、rolling/XOR hasher 和相应 memory store。
2. 实现 signed-sqrt readout。
3. 实现多尺度膨胀 causal convolution 及增量状态。
4. 定义 `qwen` 预设。
5. 实现 `GR4Channel` 和初始化策略。
6. 验证 `qwen + gr4` 的训练、prefill 和 decode。

### 阶段 5：实现 DeepSeek + mHC4

1. 实现 compressed token mapper 与持久化 token map。
2. 增加 4-gram 配置及 DeepSeek 默认 readout/postprocessor。
3. 定义 `deepseek` 预设。
4. 实现 `MHC4Channel`、Sinkhorn 和 carry 状态。
5. 验证 `deepseek + mhc4` 的训练、prefill 和 decode。

### 阶段 6：开放消融组合

至少验证以下非默认组合，确认插件边界真实有效：

- Qwen 预设去掉卷积；
- Qwen 预设加入 token compression；
- DeepSeek 预设加入 Qwen 风格卷积；
- legacy memory store 搭配 signed-sqrt readout；
- Qwen Engram 搭配 single 或 mHC4；
- DeepSeek Engram 搭配 single 或 GR4。

## 10. 验证计划与完成标准

### 10.1 legacy 兼容

- 旧 `.pth` 可以用兼容入口严格加载；
- 固定输入 logits 与迁移前一致；
- Engram 中间结果一致；
- prefill 与逐 token decode 一致；
- cache reorder 在 beam 索引下正确；
- 无 Engram 配置继续正常工作。

### 10.2 三类 Engram

- `legacy`、`qwen`、`deepseek` 均通过 shape、forward、backward 测试；
- 每个 mapper/hasher/store/readout/postprocessor 可以独立实例化；
- full-sequence hash 与 step decode hash 一致；
- padding、BOS、短序列和最大 n-gram 边界正确；
- token map 随 checkpoint 保存和恢复；
- `identity` postprocessor 不产生卷积参数或卷积 cache；
- causal conv 全序列与增量输出一致；
- Qwen 默认卷积历史长度由配置正确推导。

### 10.3 三类通道

- `single` 与原 residual add 数值一致；
- GR4 和 mHC4 输出 shape、梯度和参数注册正确；
- attention 与 FFN 始终只接收 `[B, S, D]`；
- Engram 注入始终使用 `[B, S, R, D]`；
- mHC Sinkhorn 输出有限，行和列都接近 1；
- channel state 和 Engram state 可以随 beam 一起 reorder。

### 10.4 组合与工程质量

- 三个推荐组合可以完成一次前向、反向和 optimizer step；
- 阶段 6 的消融组合均可构建和运行；
- 配置非法时给出指向具体组件的错误信息；
- `model_minigram.py` 不包含 token compression、哈希、Sinkhorn 或卷积缓存的具体公式；
- 最终模型文件数量保持为三个，不因单个组件继续扩张；
- README 或示例脚本能用最少配置切换三套展示。

## 11. 实施约束与后续工作

本次迁移遵循以下约束：

- 优先使用清晰的纯 PyTorch 实现；
- 不为了论文参数规模牺牲本地可运行性；
- 不在本阶段修改数据管线、训练循环、优化器或 GRPO 逻辑；
- 不在本阶段加入自定义 CUDA/Triton kernel；
- 不将 Qwen 与 DeepSeek 的全部训练配方误归为 Engram 组件的一部分；
- 每个迁移阶段独立提交并可回退，先保证 legacy 等价，再增加新机制。

后续可以在核心结构稳定后单独规划：

1. Muon 或其他优化器组合；
2. GRPO 阶段冻结 Engram 的实验策略；
3. 稀疏或分布式超大 memory table；
4. mHC/GR4 的融合 kernel 和性能优化；
5. 与论文规模更接近的训练复现实验。

## 12. 推荐提交序列

为降低破坏性修改风险，实际编码时建议按以下提交边界推进：

1. `test: capture legacy model and engram baselines`
2. `refactor: move legacy engram implementation into module`
3. `refactor: add pluggable engram pipeline and state`
4. `refactor: route residual flow through single channel plugin`
5. `feat: add qwen engram preset and gr4 channel`
6. `feat: add deepseek engram preset and mhc4 channel`
7. `test: cover engram overrides and cross-channel combinations`
8. `docs: document engram presets and model assembly`

每个提交都应保持项目可导入，并运行该阶段已有的全部测试。若某阶段的数值兼容失败，应在进入下一阶段前解决，避免把结构迁移和新算法误差叠加在一起。
