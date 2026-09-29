# TOML 驱动的 pretrain 与 SFT

`01_pretrain.py` 与 `02_sft.py` 已接入新版共享接口。LoRA、GRPO 仍待迁移；不兼容旧模型权重及旧 Engram 参数。

## 启动与参数

在已安装 PyTorch、Transformers、Datasets、NumPy、Tokenizers 的训练环境中执行。模型源码要求 Python 3.10+；Python 3.10 还需 `trainer/requirements.txt` 中声明的 tomli，3.11+ 使用标准库 tomllib。

```bash
python trainer/01_pretrain.py --config configs/pretrain.toml
python trainer/01_pretrain.py --config configs/pretrain.toml --batch_size 4 --epochs 1
```

CLI 仅接受 `config`、`resume_from`、`device`、`save_dir`、`batch_size`、`learning_rate`、`epochs`（以及 help）。其他参数编辑 TOML。显式 CLI 覆盖优先，未传 CLI 不覆盖文件值。必填参数缺失、未知键及错误类型会报错。

TOML 路径相对配置文件目录，CLI 路径相对当前目录。本地 tokenizer 使用 `data.tokenizer_path`，远程 tokenizer 使用互斥的 `data.tokenizer_name`；不做环境变量插值。`model.max_length` 是模型位置上限，`data.max_length` 是训练样本长度。

默认关闭 Engram；开启时默认 DeepSeek＋single，首次训练自动准备 compression。`model.engram_overrides` 是 TOML 子表，不是 JSON 字符串。GR4／mHC4 使用 `model.residual_variant` 选择。legacy 仅支持 single；单层 Engram 模型需显式设置 `engram_n_layer_list = [0]`。词表及 BOS／EOS／PAD 从 tokenizer 获取，不能在 TOML 重复指定。

`train.weight_decay` 默认 0.01。参数保持 FP32，CUDA 使用指定的 autocast 精度；非 CUDA 使用 FP32。LR、日志和保存间隔按 micro-batch 计数，`optimizer_step` 只统计实际更新。尾部累积窗口按实际批次数归一化。

## 保存与恢复

直接保存两个 `.pth` 文件，不创建 bundle、manifest、版本目录或 symlink：

- `out/minigram_pretrain.pth`：模型配置及完整 state_dict，包含 Engram 映射。
- `out/checkpoint/minigram_pretrain.pth`：额外保存 optimizer、GradScaler、下一批位置和训练设置。

`save_checkpoint(path, model)` 导出模型；传入 optimizer、scaler、step、train_config 时保存续训文件。只使用临时文件加 `os.replace`，避免写入失败破坏上一次文件。`load_checkpoint(path, model, optimizer=None, scaler=None)` 严格加载，不做宽松回退。

```bash
python trainer/01_pretrain.py --config configs/pretrain.toml \
  --resume_from trainer/out/checkpoint/minigram_pretrain.pth
```

续训按 TOML 创建相同结构，加载权重和优化器后跳到下一批。新模型自动准备 compression；续训直接恢复保存的映射。tokenizer 仍从 TOML 指定位置读取，不复制到 checkpoint，也不做内容指纹校验；请保持数据和 tokenizer 一致。

pretrain 设置 `shuffle=False`。单卡按数据集顺序读取，DDP 分配给各 rank 的样本也不打乱。没有保存 RNG：dropout 随机掩码不会从中断点接续，因此恢复训练可继续优化，但不保证与不中断训练逐步得到相同的 loss 和参数。

保存请求落在累积中途时，延迟到窗口结束；结束时保存最终权重和 checkpoint。尾部累积窗口按实际批次数归一化。恢复时检查模型配置、训练设置和批次位置；不会自动转换之前的 bundle 或旧模型裸权重。

## 分布式与验证

```bash
torchrun --standalone --nproc_per_node=2 trainer/01_pretrain.py --config configs/pretrain.toml
python -m unittest trainer.test_pretrain -v
```

只有 rank 0 写文件，不收集各 rank 的 RNG。传 `--device cpu` 可运行 CPU/Gloo；CUDA DDP 每 rank 使用一张 GPU。

测试覆盖配置解析、10 种合法组合的普通 FFN／MoE 更新、映射与严格权重加载、优化器恢复、非首 epoch 续训、尾部窗口及单 batch 保存。有 CUDA 时额外检查 BF16／FP16。旧 `tests/test_engram_smoke.py` 不属于新版测试入口。

`runtime.use_compile` 默认 false；compression 就绪检查及 MoE 动态路由可能产生 graph break。上述训练检查不代替完整的增量 decode、beam reorder 或参考算法验收。


## SFT：从新版权重开始

SFT 使用 `configs/sft.toml`，`stage="sft"`，包含 data、train、runtime、output，不接受 model 分组。模型结构、Engram 预设及映射均来自 checkpoint，加载时不重新生成 compression。tokenizer 从配置加载，词表大小及 BOS／EOS／PAD 必须与保存配置一致；不做内容指纹校验。数据长度不得超过 checkpoint 的模型位置上限。

首次 SFT 必须指定 `--init_from`，只使用模型配置与权重，重新建立 optimizer、scaler 和训练进度。`--resume_from` 只接受 SFT 完整训练 checkpoint。两者必须且只能传一个，路径相对当前目录；不提供随机初始化入口。

```bash
python trainer/02_sft.py --config configs/sft.toml \
  --init_from trainer/out/minigram_pretrain.pth
python trainer/02_sft.py --config configs/sft.toml \
  --init_from trainer/out/minigram_pretrain.pth --batch_size 4 --learning_rate 0.00005
python trainer/02_sft.py --config configs/sft.toml \
  --resume_from trainer/out/checkpoint/minigram_sft.pth

torchrun --standalone --nproc_per_node=2 trainer/02_sft.py \
  --config configs/sft.toml --init_from trainer/out/minigram_pretrain.pth
```

`data.train_on_prompt=false` 默认仅监督 assistant；设为 true 时监督全部真实 token。对话格式化、随机身份系统提示和截断逻辑保持原有行为。SFTDataset 在 SFT 入口返回独立 attention_mask，按截断后的实际长度在 padding 前生成，因此 prompt 仍参与 attention／Engram，且 PAD 与 EOS 共用 ID 时不会误屏蔽真实 EOS。默认 dataset 返回仍为 input_ids、labels，未迁移的调用方不必因这次 mask 增加而改动。

SFT 保留 shuffle 和 drop_last=True。DDP sampler 每 epoch 设置 epoch；单进程按 seed+epoch 重设 DataLoader generator。续训重建对应 epoch 的采样顺序并跳过已完成批次。没有保存 RNG，dropout 与随机系统提示不保证逐步复现。数据文件和 tokenizer 内容应保持一致。

输出默认是 `trainer/out/minigram_sft.pth`（模型导出）、`trainer/out/checkpoint/minigram_sft.pth`（完整续训状态），以及实际生效配置 `resolved_config.json`。保存采用临时文件加原子替换；累积中途的请求延迟到窗口结束，尾部窗口按实际批次数归一化。恢复校验 SFT stage、train/data 设置、world size、每 epoch 批次数及下一批位置；更换训练设置应使用 init_from 开始新的训练，而非 resume_from。

公共接口新增 `create_model_from_checkpoint(path, tokenizer)`，返回严格恢复的模型及已读取 checkpoint；`restore_training_state(state, optimizer, scaler)` 恢复其训练状态。原有 pretrain `load_checkpoint` 接口保持不变。`validate_progress` 移入 train_utils，两入口复用；没有新增通用 Trainer。
