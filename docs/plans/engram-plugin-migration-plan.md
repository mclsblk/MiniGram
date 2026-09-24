# MiniGram 模型模块重构计划：保留 legacy 算法，取消旧版兼容

## 1. 目标与实施边界

采用方案 B：新版保留 legacy、Qwen、DeepSeek 三种记忆算法，统一使用新接口；旧工程冻结保存，新版不承担旧权重、旧配置、旧 cache、旧类调用、旧 optimizer 或旧 LoRA 的兼容责任。

已确认的默认行为：

- `use_engrams` 默认仍为 `False`。
- 开启 Engram 时，默认使用 **deepseek＋single**。
- DeepSeek 映射未就绪时，允许构造模型，但 forward 明确报错；不自动降级为 identity。
- legacy 仅支持 single；Qwen、DeepSeek 可搭配 single、GR4、mHC4。关闭 Engram 时三种通道均可使用。

本轮只修改模型与文档。训练、推理、checkpoint 和 LoRA 工具的接入放到以后；旧脚本在新版中的可用性不作承诺，文档明确这一限制。

实施控制保持不变：

- 先在基线提交 `2b7125e88906cbe1e1e5dd803f0e1651860c44aa` 建立本地冻结分支 `codex/legacy-v1` 和 tag `minigram-legacy-v1`。
- 每阶段独立本地提交，汇报后等待用户明确确认下一阶段。
- 暂不 push、不通知远端任务、不安装依赖、不运行模型数值验证。
- 出现算法取舍、接口变更或范围扩张时，先询问用户，不自行修改已确认决策。

## 2. 模块结构与算法范围

### 2.1 文件划分

采用四文件结构：

```text
model/
├── model_minigram.py   # 基础网络组件、层级连接、主模型组装
├── engram.py           # 五段记忆管线、预设、状态、映射 helper、集中 builder
├── channels.py         # single、GR4、mHC4 及集中 builder
└── common.py           # 确实共享的基础组件
```

**暂不新增独立 legacy 文件。**旧 Engram 约 140 行，拆出共用查表、状态与组装后，其专属逻辑集中维护在 `engram.py` 的 legacy 区段即可。不保留完整旧模型副本，也不创建旧 API wrapper。

依赖保持单向：公共组件不反向导入主模型，Engram 与通道不依赖主模型的运行时定义。不同算法的 norm 参数化及 dtype 语义分别保留，不因共享文件而强行统一。

继续使用普通函数、轻量 dataclass 和 `nn.Module`。不引入动态注册表、插件发现、抽象类体系、多级 factory 或泛化上下文。

### 2.2 Engram 管线

保留五个组件插槽：

```text
token_mapper → hasher → memory_store → readout → postprocessor
                                           ↑
                                完整 residual streams
```

内部入口接收 `input_ids`、完整 `[B,S,R,D]` streams、可选 mask 和 `EngramState`；返回同形状的 `delta` 及更新后的状态。

- 门控读取各条 stream，不先压缩成单一 hidden。
- Engram 完成逐流门控；通道只将 delta 加回 streams，不再追加写入门。
- separate、packed 两种存储均实现且可切换，明确 head 顺序、各 head 容量、padding bucket 与 packed offset。
- legacy 默认 separate；Qwen、DeepSeek 默认 packed。不提供已有 checkpoint 的跨布局转换器。
- 删除额外的“多阶输出可学习加权”readout；保留参考算法自身需要的逐维 q/k 权重。
- full、分块 prefill、逐 token decode 共用同一份哈希与卷积公式。

| 预设 | 保留的算法机制 | 默认位置 |
| --- | --- | --- |
| legacy | 旧哈希、独立表布局、旧门控及归一化、旧卷积残差行为 | attention 后 |
| qwen | 2/3-gram、原 token、参考 signed-sqrt 门控、膨胀 depthwise 因果卷积 | attention 前 |
| deepseek | 2/3/4-gram、参考 token compression、参考 signed-sqrt 门控、identity 后处理 | attention 前 |

legacy 保留旧公式，但采用新参数组织、配置、状态和返回值。不要求旧整模型逐元素一致，也不安排跨版本数值对照。

Qwen 卷积保留参考的 norm、SiLU 和残差组合；默认 kernel 为 4、dilation 为 3，历史长度按公式推导，不额外建设多尺度并联结构。

参考来源锁定为当前本地 Qwen／DeepSeek 源码快照，具体 SHA-256 与对应函数见第 6 节。算法存在的边界、归一化、哈希生成差异按参考保留。

## 3. 新接口、配置和状态

### 3.1 统一配置

三种预设统一通过 `engram_variant` 与浅层 `engram_overrides` 选择和调整，不再单独保留 legacy 的旧参数解析规则。

保留公共控制字段：

- `use_engrams`：是否启用记忆。
- `engram_variant`：默认 `deepseek`。
- `engram_overrides`：已知组件及算法参数的浅层覆盖。
- `engram_n_layer_list`：插入层列表，默认 `[1]`。
- `residual_variant`：默认 `single`。
- `residual_low_rank`：GR 的低秩维度。

旧 Engram 容量、算法字段不再自动映射；发现这些已废弃字段时明确报错并指向 overrides。没有旧配置兼容路径或旧权重自动识别逻辑。

默认规模与校验：

- bucket 基数 1024，每阶 4 heads。
- head dimension 默认按 `ceil(hidden_size / 总memory_heads)` 计算，允许显式覆盖。
- GR rank 默认 `min(64, hidden_size)`。
- 通道数由 variant 决定为 1 或 4。
- 校验按组件执行，不向 legacy 添加算法本身不需要的整除条件。
- 未知覆盖字段、不支持的组合、非法容量或插入层在构造时明确报错。
- 保存解析后的有效配置，支持新版配置自身的序列化往返；只承诺新版配置与新版权重配套恢复。

### 3.2 通道接口与归一化

统一采用显式读取和写回：

```text
initialize(hidden) → ChannelState
read(state, branch) → hidden, branch_context
子层计算(hidden) → branch_output
write(state, branch_output, branch_context) → ChannelState
inject(state, delta) → ChannelState
finalize(state) → hidden
```

- attention、FFN 始终接收 `[B,S,D]`。
- 通道参数按层、按分支独立注册；最终 mixer 单独注册。
- `branch_context` 保存本次写回系数和下一分支信息，不藏在 module 的可变成员里。
- single 使用标准 residual add；GR 遵循参考的分组归一化、动态读写和最终 mixer；mHC 遵循参考的 collapse、norm、post/comb 顺序。
- 不把旧模型的 norm 无条件叠加到 GR 的归一化读出之后。

mHC 的 pre 沿模型深度延迟使用：attention 使用上一分支留下的 pre，本分支生成的 pre 给下一分支，最终 pre 用于模型末端归并。每次 forward 重新初始化通道状态，不跨 decode 调用复用上一 token 的 pre。

mHC 使用 FP32 Sinkhorn，默认 20 次迭代、`eps=1e-6`。初始化采用近恒等残差与均衡读出：pre 约 `1/4`、post 为 `1`、comb logits 对角为 `8`、非对角为 `0`，动态 scale 为 `0.01`。这是 MiniGram 的初始化选择，不声称复现原模型训练初始化。特殊初始化不得被顶层初始化流程覆盖。

### 3.3 Token map 与推理状态

token compression 仅实现参考 tokenizer 文本归一化算法，提供离线 helper 与模型级 setter；不实现语料统计映射或任意自定义映射实验。

- 允许先构造模型，再注入或加载映射。
- 持久化 token map，并同步建立或恢复依赖压缩词表大小的哈希信息。
- 首次 forward 前必须完成映射准备；不支持计算过程中更换映射。
- 映射就绪后，推理无需重新处理 tokenizer。

`EngramState` 保存有限 token 历史和后处理历史，并实现 beam batch 重排。模型只使用新的 `engram` cache 字段，不读取或转换旧 `engram_tail/engram_conv` 格式。

`ChannelState` 是一次 forward 内沿层传播的计算状态；`EngramState` 才是跨 decode 调用的历史。二者不能混用。

## 4. 分阶段实施

取消“先建立旧兼容层，再改造”的路线，改为先确定新契约、建立基础层，再分别接入算法。

| 阶段 | 工作 | 阶段检查重点 |
| --- | --- | --- |
| 0 | 冻结旧分支和 tag，更新计划，记录参考指纹、支持范围及接口 | 删除旧兼容承诺，明确旧脚本接入后置 |
| 1 | 建立公共组件、新配置、状态及集中 builder 边界 | 单向依赖、配置序列化和组件 shape 契约 |
| 2 | 实现 single，改造主模型 residual 流和归一化连接 | 无 Engram 路径、attention／FFN 输入、模型最终输出 |
| 3 | 将 legacy 算法接入五段管线和新状态 | 旧公式逐项审查；仅支持 single；不添加兼容入口 |
| 4 | 独立实现 GR4，先审查无 Engram 路径 | 分层参数、读写系数、归一化、初始化与最终 mixer |
| 5 | 独立实现 mHC4，先审查无 Engram 路径 | delayed pre、Sinkhorn 方向、状态寿命及初始化 |
| 6 | 实现新哈希／存储／读出组件和 Qwen 卷积，组装 Qwen | 逐流门控、表布局、卷积历史和各通道接口 |
| 7 | 实现参考 token map 与 DeepSeek 预设 | 映射生命周期、哈希依赖、4-gram 及无卷积路径 |
| 8 | 完成允许的交叉组合和模型使用文档 | 新 API 示例、错误提示、后置验证命令及入口迁移说明 |

每阶段仅执行本机可用的 AST 解析、`git diff --check` 和静态审查，然后形成独立提交。汇报必须分别列出“已实现”“静态检查结果”“待运行验证”，不能将静态通过表述为算法验收通过。

## 5. 后置验证与完成标准

运行验证统一后置，命令保存在文档中，不新增测试文件或专用验证框架。

验收覆盖：

- **legacy**：审查旧哈希、门控、归一化及卷积公式；验证新版内部 forward/backward、full/decode 自洽，不验证旧权重加载或跨版本逐元素一致。
- **通用组件**：separate／packed 在等价表内容下检索一致；非零卷积权重下 full/decode 一致；identity 后处理无卷积参数和历史。
- **边界与状态**：短序列、各阶 n-gram 起点、EOS、padding/mask、分块 prefill、逐 token decode，以及含重复索引的 beam reorder。
- **映射与保存恢复**：未就绪映射报错；setter 与新版权重加载后恢复一致；新版配置与权重往返恢复。
- **通道**：三种通道先独立验证，再与记忆组合；检查参数注册、有限梯度、GR 动态读写、mHC pre 时序、Sinkhorn 行列和及最终归并。
- **组合**：legacy＋single，以及 Qwen／DeepSeek 与三种通道；检查去卷积、加参考 compression、增加 Qwen 卷积及存储布局切换。legacy＋多通道必须明确拒绝。
- **训练可执行性**：推荐组合各完成一次 forward、backward 和 optimizer step；覆盖普通 FFN 与现有 MoE 路径。

FP32 full/decode 比较默认 `atol=1e-5、rtol=1e-4`；哈希 ID 必须完全一致；Sinkhorn 行列和误差不超过 `1e-4`。超出阈值先定位，不自动放宽。其他精度在选定验证环境后单独确认。

旧版冻结用于保留历史工程；新版验收只针对新模型契约和所保留的算法机制。模型模块运行验收通过后，再另行规划训练、推理与 checkpoint 脚本接入。

## 6. 参考源码快照与定位

以下路径相对仓库根目录，均指向本地参考材料，不是本轮待修改文件。SHA-256 用于检查材料是否变化；它不是上游 Git revision，也不意味着这些参考材料已随本仓库提交。参考材料发生变化时应先核对差异，不自动改用更新版本。

| 本地文件 | 参考内容 | SHA-256 |
| --- | --- | --- |
| `DeepSeek-V4.1-Flash-source/modeling_qwen4_exp.py` | `Qwen4ExpTextRMSNorm`；`Qwen4ExpTextGatedResidual`；`Qwen4ExpTextNGramEmbedding`；`Qwen4ExpTextPLELayer`；decoder／model 的注入、读写和最终 mixer 顺序 | `797a18fd6dd76c574d237a5643759acdeb1c4d0f1c2508693f8fdabce0a19057` |
| `DeepSeek-V4.1-Flash-source/configuration_qwen4_exp.py` | Qwen 的 n-gram、卷积、GR 参数和结构约束；大模型容量不直接照搬 | `b78132d8cd935437208ee281fa4569b771a63fcb58ebffe84f3e62f5b86235ca` |
| `DeepSeek-V4.1-Flash-source/inference/engram.py` | `build_compressed_token_map`；`compute_hash_multipliers`；`EngramLayout`；`NgramHashState` | `11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897` |
| `DeepSeek-V4.1-Flash-source/inference/model.py` | `Engram`；`Block.hc_mixes/hc_pre/hc_post/forward`；`make_identity_pre_mix`；模型的注入和最终归并 | `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65` |
| `DeepSeek-V4.1-Flash-source/inference/kernel.py` | `hc_split_sinkhorn_kernel` 和 `hc_split_sinkhorn` 中 pre／post／comb 公式、矩阵方向和迭代顺序；只参考算法，不引入 kernel 依赖 | `1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455` |

legacy 的算法依据是基线提交中的 `model/model_minigram.py`：`EngramModule` 及 `TransformerBlock` 的 Engram 归一化和插入顺序。只继承算法，不继承旧接口契约。

### 6.1 可重跑的静态检查命令

在仓库根目录执行；不导入模型，也不需要 PyTorch：

```sh
git diff --check
python3 - <<'PYTHON'
import ast
from pathlib import Path

files = sorted(Path('model').glob('*.py'))
for path in files:
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
print(f'AST parsed: {len(files)} model files')
PYTHON
```

检查冻结引用：

```sh
git rev-parse codex/legacy-v1 minigram-legacy-v1
```

两行均应为 `2b7125e88906cbe1e1e5dd803f0e1651860c44aa`。后续运行验证命令在模型接口落地后按第 5 节补齐，阶段 0 不将尚不存在的接口示例标记为可运行或已验证。

## 7. 阶段进度

- **阶段 0 已落实**：建立本地冻结分支与 tag；以方案 B 替换旧计划；记录新接口、支持范围、参考指纹与静态检查命令。阶段 0 未修改模型实现。
- **阶段 1 已落实**：新增公共 norm、解析后的 EngramSpec、新配置校验、显式状态和集中 builder 入口；算法构建尚未开放，详见第 8 节。配置序列化往返和状态重排仅完成实现及静态审查，未运行验证。
- **阶段 2／3 已合并编码交付**：按用户要求合并为一次本地提交；接入 single 通道、legacy 五段管线和新 cache。用户明确要求 coding 后不审查，因此本次未执行代码审查、AST 检查、diff 检查或运行验证。
- **阶段 4～8 未开始**：需用户明确确认下一阶段后继续；每阶段完成后更新本节。
- **运行验证全部待执行**：当前没有算法、数值、梯度、保存恢复或增量推理通过的结论。

## 8. 阶段 1：接口记录与当时的中间版本限制

### 8.1 公共组件与依赖

- `common.py` 提供 `RMSNorm` 和 `QwenRMSNorm`。前者保留 MiniGram 的先回转 dtype 再乘权重顺序；后者使用零中心权重、可选分组和 FP32 乘权重后回转 dtype。
- `engram.py` 提供配置解析、`EngramSpec`、`EngramState` 和 `build_engram_layers()`。
- `channels.py` 提供 `ChannelState`、`BranchContext` 和 `build_residual_channel()`。
- 主模型导入上述基础接口；三个新文件均不导入主模型。配置解析函数只产生数据，不构造模块；每类插件仍只有一个模块构建入口。

### 8.2 新配置字段

`MiniGramConfig` 显式接收公共开关、variant、overrides、插入层列表和 GR rank。`engram_overrides` 在构造配置时展开为完整的有效字段；配置保存普通字典和列表，不保存 dataclass、模块或 Tensor。派生的 `residual_channels` 为 1 或 4，重新读取配置时检查其与 variant 一致。

以下是三种预设共用的浅层 overrides，组件名直接标明参考计算的差异：

| 字段 | 可选值或含义 |
| --- | --- |
| `ngram_orders` | 非空、严格递增、至少为 2 的整数列表；legacy／qwen 默认 `[2, 3]`，deepseek 默认 `[2, 3, 4]` |
| `token_mapper` | `identity`、`compressed` |
| `hasher` | `legacy`、`qwen_xor`、`deepseek_xor`；后两者的 multiplier／seed 生成不能混同 |
| `memory_store` | `separate`、`packed` |
| `readout` | `legacy`、`qwen_signed_sqrt`、`deepseek_signed_sqrt`；保留各自 norm 和门控语义 |
| `postprocessor` | `identity`、`legacy_conv`、`causal_conv` |
| `insertion` | `before_attention`、`after_attention` |
| `bucket_size`、`num_heads`、`head_dim` | bucket 基数、每阶 head 数、每 head 向量宽度；默认按第 3.1 节解析 |
| `hash_seed` | legacy 默认 17；Qwen 默认参考值 1234；DeepSeek 为 `None`，按参考从层号派生 seed，不接受人为覆盖 |
| `conv_kernel_size`、`conv_dilation` | legacy 默认 3／1；Qwen causal_conv 默认 4／最大 n-gram 阶数；identity 默认 1／1 且后续不会创建卷积 |

旧字段 `engram_vocab_size`、`engram_n_gram_list`、`engram_num_heads`、`engram_conv_size`、`engram_hash_seed` 一律报错并给出 overrides 中的替代字段，不提供自动映射。参数类型遵循接口约定，不逐项检查 Python 精确类型；未知组件、重复或乱序 n-gram、错误通道数等仍在构造配置时拒绝。

插入层使用零基索引，保存时排序；始终校验非负和去重，启用 Engram 时检查层号上界。未启用时保留默认 `[1]`，不会因此阻止单层无 Engram 模型。legacy 预设、legacy readout 或 legacy convolution 启用时均仅用于 single。

### 8.3 状态与 builder 契约

- `EngramState.hash_tail` 为 `[B,T]` 的哈希域上下文，包含对应算法的边界标记；`post_state` 为可选 `[B,T,R,D]` 卷积输入历史。`reorder(beam_idx)` 返回新的状态，按各 tensor 所在设备执行 batch 索引，不修改原状态。
- `ChannelState.streams` 为 `[B,S,R,D]`，`pre_mix` 为可选 FP32 `[B,S,R]`。它们不是 decode cache。
- `BranchContext` 只保存 `post_mix`、`comb_mix` 和 `next_pre_mix`。single 均不用，GR 使用 post，mHC 使用三者；由一次 read 产生，紧接着交给对应 write。
- 通道 `branch` 标识为 `(layer_index, "attention" 或 "ffn")`，因此集中通道模块能够持有各层各分支独立参数，而不是全模型共享一份映射。
- `build_engram_layers(config)` 的返回类型为按层号字符串索引的 `nn.ModuleDict`；禁用 Engram 时返回空模块集合。`build_residual_channel(config)` 是唯一通道模块构建入口。具体算法在后续阶段接入这些入口。

### 8.4 阶段 1 当时的可达路径与待运行验证

本小节记录阶段 1 的历史状态；阶段 2／3 后的可用范围以第 9 节为准。

本阶段保留主模型原有的 **关闭 Engram＋single** 直连路径，尚未改成通道 API。构造配置与实例化算法是两件事：配置可以描述所有目标预设，但本阶段尝试在模型中启用任何 Engram、GR4 或 mHC4 都会显式抛出 `NotImplementedError`，并提示对应实施阶段，不会静默退化。

旧 `EngramModule` 定义暂留主文件作为阶段 3 的公式迁移来源。它已不在本阶段顶层模型的可达构造路径中，不属于新版 API，也没有通过补回旧配置字段让它继续工作。阶段 3 将用新管线替换该定义及旧连接代码。

当前源码已提供配置的普通数据序列化结构和状态 reorder 实现，但以下运行检查仍后置：

- `MiniGramConfig.to_dict()`／JSON 保存后重新构造是否保留完整有效配置及自定义 RoPE 设置。
- 各非法配置是否按预期报错、禁用 Engram 时是否不创建记忆参数。
- `EngramState.reorder()` 对重复 beam 索引和设备位置的行为。
- 公共 norm 的 dtype、分组、参数初始化和数值行为。
- 后续阶段接入后，无 Engram 主模型、五段管线和通道生命周期的运行行为。


## 9. 阶段 2／3：合并编码交付

用户授权合并阶段 2／3，并要求 coding 后不进行审查。本次仅修改 `model/channels.py`、`model/engram.py`、`model/model_minigram.py` 和本文档；未修改公共组件、训练脚本、推理脚本、工具、依赖或测试文件。第 4 节原定的阶段后检查在本次交付中不执行，不表示这些检查通过。

### 9.1 已接入的模型路径

- `SingleResidualChannel` 实现 initialize、read、write、inject、finalize。原 attention／FFN 前的 RMSNorm 按层、按分支归属到通道中，末端 RMSNorm 归属到 finalize；没有新增残差映射参数。
- 主模型只在顶层注册 channel 和 Engram ModuleDict，调用 block 时传递引用；block 不重复注册这些模块。四维 ChannelState 沿层传播，attention 和 FFN 接收通道读出的三维 hidden。
- 删除主文件中的旧 `EngramModule` 和旧 Engram 连接代码。legacy 的 query norm 归属 readout；插入前后通过构造时解析的布尔标记选择，block 不包含具体记忆算法。
- 每个 legacy 层依次执行 IdentityTokenMapper、LegacyHasher、MemoryStore、LegacyReadout、PostProcessor，返回 delta 和新 EngramState。
- 存储支持 separate／packed，均按阶数再按 head 排列。legacy 保留各 head 的零号 padding bucket；packed 在每 head 对应位置保留零行，并通过输出 mask 阻断这些行的梯度。
- 后处理支持 legacy_conv 和 identity；identity 不创建卷积参数，返回的 post_state 为 None。legacy 卷积的历史为 `[B,T,1,D]`，长度由 kernel 推导。
- legacy 继续按原 token IDs 计算哈希，EOS／padding 不被重新解释为分段标记；attention mask 不改变 legacy 哈希和卷积公式。后续 Qwen／DeepSeek 的边界机制不在本阶段实现。
- 顶层 post_init 后恢复 legacy 的零卷积初始化及表中 padding 零行；没有权重名称迁移或旧接口 wrapper。

### 9.2 新 cache 约定

`past_key_values` 为逐层字典序列，每层只接受 `attn` 和可选 `engram`：

- `attn` 为 `(key, value)` tuple。
- `engram` 为 EngramState，包含有限 hash_tail 与可选卷积历史。
- 拒绝旧 cache key 和非约定的外部 cache 容器，不自动转换。
- 有历史输入时要求 `use_cache=True`，层数必须与模型一致；关闭 cache 时模型返回 `past_key_values=None`。
- beam 重排分别处理 attention tensor 和 EngramState，不保存或重排跨 forward 的 ChannelState。

### 9.3 当前配置示例

以下为已编码接口的用法说明，**未执行验证**：

```python
from model.model_minigram import MiniGramConfig, MiniGramForCausalLM

config = MiniGramConfig(
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_kv_heads=2,
    intermediate_size=64,
    vocab_size=64,
    max_length=32,
    dropout=0.0,
    flash_attention=False,
    use_engrams=True,
    engram_variant="legacy",
    residual_variant="single",
    engram_n_layer_list=[1],
    engram_overrides={"bucket_size": 17},
)
model = MiniGramForCausalLM(config)
```

通过 `engram_overrides={"memory_store": "packed", "postprocessor": "identity"}` 可以选择 packed 表和无卷积 legacy 变体。关闭 Engram 时返回空记忆模块集合；GR4／mHC4 和 Qwen／DeepSeek 仍明确报未实现。默认 Engram 预设仍为 DeepSeek，因此目前启用记忆时必须显式选择已实现的 legacy。

### 9.4 交付状态

本次为未经编码后审查和验证的 coding 交付。shape、梯度、full/decode、beam reorder、初始化、模型保存恢复及旧算法公式迁移的正确性均未在本阶段验证，不作已通过声明。后续验证范围继续使用第 5 节，不增加跨版本兼容验收。


## 10. 配置与 cache 校验精简

本次只清理阶段 1 和阶段 2／3 中重复或无实际作用的防御代码，不改变算法公式、五段管线或通道架构。

- 有效 Engram 配置由 MiniGramConfig 统一解析和校验；builder 直接使用已解析字段构造 EngramSpec，不再次解析，也不重复检查 legacy 与 single 的组合。
- 删除逐字段的 Python 精确类型检查，保留容量、阶数、插入层等算法取值约束。ngram_orders 仍要求非空、各阶至少为 2、严格递增且不重复，不通过排序去重静默修正输入。
- GR rank 的正值约束只对 GR4 生效；卷积 kernel／dilation 的正值约束只对启用卷积的后处理生效。
- cache 格式在模型入口校验，内部读取直接遵循新 cache 契约；删除 EngramLayer 的重复状态类型检查及内部 helper 的格式兜底。原 normalize helper 改名为 validate，明确其不负责转换。

保留未知 overrides、废弃字段、不支持组合、cache 层数和 delta 精确形状等检查。residual_channels 的序列化及一致性约束保持现状，本次不扩展配置接口调整范围。

按本次交付约定，清理完成后不执行编码后审查、AST 解析、git diff --check 或运行验证。这里只记录代码变更，不表示算法或运行验收通过；阶段 4 及后续阶段尚未开始。


## 11. 校验边界集中维护

经用户确认，新增 `model/validation.py`，集中维护模型配置、Engram 配置及组合约束、外部 cache 容器校验。原四个模型模块的计算职责不变；新增文件只承担边界校验。

依赖方向固定为主模型 → Engram → validation，以及主模型 → validation。validation 不导入主模型、Engram 或 channels，不引用具体状态类，只读取普通配置字段和外部 cache 容器；不解析默认值、不修改参数、不构造组件。

主模型配置入口调用模型配置及组合校验，Engram resolver 在解析依赖默认值前后分别调用对应配置校验。forward 入口集中检查 cache 容器、层数及 use_cache；beam reorder 入口复用同一函数。删除 EngramState 的具体类检查，内部直接按状态接口访问字段和 reorder 方法。

本次为职责迁移及具体类检查删除，不宣称迁移校验代码减少了总体代码量。算法分支、未实现组件的显式错误和通道 delta 形状约束不变。按交付约定，不执行编码后审查、静态检查或运行验证，不推进阶段 4。


## 12. 阶段 4：GR4 编码交付

GR4 已接入现有通道接口。每层 attention／FFN 分别注册 GRMixer，末端单独注册无写回投影的 final_mixer。按 Qwen 参考使用分组零中心 RMSNorm、低秩 SiLU 读门、逐维 sigmoid 读系数及跨流 mean；写系数为 `2 * sigmoid(proj(normalized) / 4)`。attention／FFN 接收 `[B,S,D]`，不再叠加 single 的 RMSNorm，末端 mixer 后也不追加 norm。

GR 线性参数按 initializer_range 正态初始化，零中心 norm 权重为零；顶层 post_init 后通过通道统一初始化入口恢复。关闭 Engram 时可选择 gr4；legacy＋gr4 仍拒绝，Qwen／DeepSeek 尚未接入。

本次与阶段 5 连续施工，分别本地提交；沿用直接编码交付约定，不执行编码后审查、静态检查或模型运行验证。动态读写、参数注册、梯度、初始化、full/decode 及普通 FFN／MoE 路径仍待运行验证。


## 13. 阶段 5：mHC4 编码交付及当前可用范围

mHC4 已接入相同通道接口。每层 attention／FFN 分别注册动态投影、base、scale 及 DeepSeek 风格 RMSNorm；末端单独注册 final_norm。公共组件新增 DeepSeekRMSNorm，保留先在 FP32 归一化并乘权重、再转换 dtype 的参考顺序，不改变 single 或 GR 的 norm。

每次 forward 扩展四条 streams 并新建均衡初始 pre（各 1/4）。分支先从完整 streams 生成系数，再用上一分支的 pre collapse、norm、运行子层；写回按 comb 的 `[source, destination]` 方向混合残差，加上 post 加权子层输出，并保存本分支生成的 pre。最后一个 pre 用于末端 collapse。ChannelState 不进入 decode cache；Engram inject 保留当前 pre。

系数投影与 Sinkhorn 显式关闭 autocast 并使用 FP32。按参考先 row softmax 加 eps、再归一化列，随后执行 19 轮行列归一化，总计 20 轮，eps 为 1e-6。pre 使用 sigmoid 加 eps，post 使用两倍 sigmoid。MiniGram 初始化为 pre base `-log(3)`、post base 0、comb 对角 logits 8／非对角 0、动态 scale 0.01；动态投影按 initializer_range 正态初始化，因此 pre／post 是约 1/4／约 1。顶层 post_init 后恢复这些初始化。此处不声称复现参考训练初始化。

当前关闭 Engram 时可选 single、gr4、mhc4；开启 Engram 仍只可选 legacy＋single。Qwen／DeepSeek 实现及其多通道接入留待阶段 6／7。第 9 节记录的 GR4／mHC4 未实现状态已被本节替代。

以下为后置验证入口示例，尚未执行；按第 5 节继续覆盖 full/decode、梯度、参数保存恢复、Sinkhorn 行列和、delayed pre 及初始化：

```bash
python - <<'PYCODE'
import torch
from model.model_minigram import MiniGramConfig, MiniGramForCausalLM
for variant in ("gr4", "mhc4"):
    for moe in (False, True):
        config = MiniGramConfig(
            hidden_size=32, num_hidden_layers=2, vocab_size=64,
            num_attention_heads=4, num_kv_heads=2, intermediate_size=64,
            max_length=32, dropout=0.0, flash_attention=False,
            use_engrams=False, residual_variant=variant, use_moe=moe,
        )
        model = MiniGramForCausalLM(config)
        tokens = torch.randint(0, config.vocab_size, (2, 8))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        output = model(tokens, labels=tokens, use_cache=False)
        loss = output.loss + output.aux_loss
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        print(variant, moe, float(loss.detach()))
PYCODE
```

本阶段仅编码交付，未进行编码后审查、AST 解析、git diff --check 或运行验证；数值与算法验收尚未完成。未修改训练／推理脚本、依赖或测试文件，未推进阶段 6。


## 14. 阶段 6：Qwen Engram 编码交付

Qwen 预设已组装进五段管线，支持 single、GR4、mHC4，默认 packed，允许切换 separate 或使用 identity 去除卷积。本阶段仅修改 engram.py 与本文档，未修改通道、主模型、训练／推理入口或依赖；DeepSeek 及 compression 仍留待阶段 7。

### 14.1 哈希与存储

QwenHasher 按本地参考 `_build_layer_multipliers`、`_splitmix64` 和 `_shift_right_ignore_eos` 的公式生成哈希。缺失前缀用 EOS 补齐，EOS 后的 token 不读取上一段；EOS 自身仍能读取它之前的同段 token。多 EOS 配置按参考使用列表首项。padding mask 不重写 token ID 或重置哈希历史。

head 顺序为 n-gram 阶数再 head 索引。各 head 容量按 bucket 基数起的连续质数分配；全局 head 序号和 multiplier seed 使用排序后的 Engram 插入列表内序号，而非模型绝对层号。实际容量和 multipliers 保存为 buffer。Qwen 不预留零号 padding bucket；legacy 继续保留。packed offset 是各 head 容量的前缀和，separate 使用同一组容量；不额外分配参考大模型用于 embedding 对齐的尾部空行，也不提供跨布局权重转换器。

哈希状态保存最多 max(ngram_orders)-1 个原 token。full、分块 prefill 和 decode 均调用同一个 shift／XOR 过程；没有单独 decode 近似公式。

### 14.2 逐流读取与卷积

QwenReadout 将记忆投影到逐流 key 和共享 value，query/key 按流使用 Qwen 零中心 RMSNorm。每条流独立执行点积、除 sqrt(D)、signed-sqrt（abs clamp 至 1e-6）和 sigmoid；不压缩 streams，也不额外增加多阶输出权重。

QwenCausalConv 保留 norm → depthwise causal conv → SiLU，再加回原门控值的顺序。默认 kernel=4、dilation=3，历史长度为 `(kernel-1)*dilation`，默认 9；缓存为已归一化且应用 mask 的 `[B,T,R,D]` 卷积输入。mask 同时作用于门控残差与卷积输入；按参考不再对卷积输出补 mask，EOS 不重置卷积。卷积采用统一左 padding／切片公式处理所有调用长度。

identity 后处理不创建卷积参数或历史；也不额外引入 Qwen 卷积组件的 mask 操作。顶层特殊初始化入口恢复 Qwen norm 的零权重与卷积零权重，不改变其他线性层的初始化。

### 14.3 使用与后置验证

```python
config = MiniGramConfig(
    use_engrams=True,
    engram_variant="qwen",
    residual_variant="gr4",  # 也可使用 single 或 mhc4
    engram_n_layer_list=[1],
)
# 可选覆盖：{"memory_store": "separate"} 或 {"postprocessor": "identity"}
```

当前可编码构造的组合为 legacy＋single、Qwen＋single／GR4／mHC4；默认 engram_variant 仍为 deepseek，启用时暂须显式选择 legacy 或 qwen。本节替代前面交付记录中 Qwen 未实现的历史状态。

后置运行命令可复用第 13 节，将 residual_variant 循环增加 single，并设置 use_engrams=True、engram_variant="qwen"。除 forward／backward／optimizer step 外，仍须按第 5 节验证：哈希 ID 完全一致；含 EOS、短序列和 padding 的 full／分块／decode；非零卷积权重下 FP32 输出容差 atol=1e-5、rtol=1e-4；重复 beam 索引；同表内容的 separate／packed；identity 无卷积参数及历史；三通道普通 FFN／MoE 和新版权重保存恢复。

本次按直接交付约定不进行编码后审查、AST 解析、git diff --check 或数值验证。参考文件在施工前核对 SHA-256，与第 6 节一致；这不是实现验收。未 push，未推进阶段 7。


## 15. 阶段 7：DeepSeek 与 token compression 编码交付

经用户确认，缺失历史使用模型 `config.pad_token_id`，不新增 setter 参数或 overrides 默认值。映射准备前必须设置该字段；缺失时 setter 明确报错。DeepSeek 默认仍是 compressed＋deepseek_xor＋packed＋deepseek_signed_sqrt＋identity，开启 Engram 后可先构造模型，但映射未就绪的 forward 明确报错，不降级为 identity。

### 15.1 离线映射与持久化

`model.engram.build_compressed_token_map(tokenizer)` 沿用参考 tokenizer 文本归一化：NFKC、NFD、去重音、小写、空白合并与首尾处理，并用私用字符保留单空格 token；含 Unicode replacement character 的部分 UTF-8 token 按原 token 形式分组。返回完整 lookup 和压缩词表大小，不引入语料统计算法。helper 按需使用现有环境的 tokenizers，本轮未安装依赖。

模型级 `set_engram_token_map(token_map)` 接收该 helper 的整数映射，校验长度、连续压缩 ID 和 config.pad_token_id 后，同步配置各层查表、压缩词表大小、压缩 pad ID 及哈希乘数。校验集中在 validation.py，仍不引用具体模型类。setter 不是自定义映射实验 API；文档只支持参考 helper 的结果。

每个 compressed mapper 使用长度等于 config.vocab_size 的持久化 long buffer，尚未准备时为 -1，压缩词表大小为 0。各 hasher 的乘数、容量和 DeepSeek pad ID 也持久化为固定形状 buffer。新配置配套的新版权重可直接恢复，不需要再次读取 tokenizer 或根据变长 buffer 修改加载逻辑。首次 forward 后拒绝用 setter 更换映射；更换 tokenizer／加载另一套映射权重应创建新模型，不复用已有 decode 历史。

Qwen＋compression 的 hash multipliers 和 EOS ID 同步进入压缩域，并持久化恢复。未压缩的 Qwen／legacy 路径不需要 setter。

### 15.2 DeepSeek 公式

DeepSeekHasher 默认计算 2／3／4-gram。乘数按参考 `np.random.default_rng(10007 * layer_id)` 生成奇数，边界由压缩词表大小确定；不以 PyTorch RNG 替代 NumPy，避免地址变化。各层各阶各 head 使用不重复质数容量，默认 packed 不预留零号 padding bucket。

短序列缺失历史和 mask=False 的 token 按参考用压缩 pad ID 填充；mask=False 以 DEAD=-1 保存到历史，并阻断更早回看。EOS 不单独截断。历史最多保留 max(ngram_orders)-1 项；full、分块和 decode 使用相同逐 lookback XOR 公式。

DeepSeekReadout 使用联合 key/value 投影和各 stream 的逐维 q_weight／k_weight，按参考在 FP32 中计算独立 RMS 统计、点积和 copysign signed-sqrt，再 sigmoid 门控共享 value。返回 delta，由通道执行一次残差写回；mask=False 关闭当前门控。默认 identity 不创建卷积参数或 post_state，q/k 初始化在顶层初始化后恢复为 1。

### 15.3 新接口示例与待验证

```python
from model.engram import build_compressed_token_map
from model.model_minigram import MiniGramConfig, MiniGramForCausalLM

# tokenizer 已由调用方准备；这里不改变训练／推理脚本。
config = MiniGramConfig(
    vocab_size=len(tokenizer),
    pad_token_id=tokenizer.pad_token_id,  # 必须有明确值
    use_engrams=True,
    engram_variant="deepseek",
    residual_variant="single",  # 也支持 gr4、mhc4
)
model = MiniGramForCausalLM(config)
token_map, compressed_vocab_size = build_compressed_token_map(tokenizer)
model.set_engram_token_map(token_map)

# 后置验证命令片段：在选定验证环境执行，不是本轮执行结果。
model.save_pretrained("/tmp/minigram-stage7")
restored = MiniGramForCausalLM.from_pretrained("/tmp/minigram-stage7")
# restored 无需重新执行 tokenizer helper 或 setter。
```

当前三预设均已编码接入；legacy 仍仅允许 single，Qwen／DeepSeek 可接三种通道。现有 overrides 可组合已实现组件，组合覆盖与入口使用文档仍属于阶段 8。此前各阶段记录中 DeepSeek 未实现的描述为历史状态，由本节替代。

待验证范围包括：未准备映射报错、pad_token_id 缺失报错、setter 与保存恢复一致、首次 forward 后 setter 拒绝更新、mask／短序列／4-gram 的哈希 ID、full／分块／decode、重复 beam reorder，以及三通道下普通 FFN／MoE 的梯度与 optimizer step。输出阈值沿用第 5 节，不放宽。

本次仅编码交付，未进行编码后审查、AST 解析、git diff --check 或模型数值验证。施工前参考指纹与第 6 节一致。本轮不安装依赖，不修改训练／推理脚本，不 push，不推进阶段 8。
