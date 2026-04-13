from transformers import PretrainedConfig
from typing import Optional

import torch
import math

class MiniGramConfig(PretrainedConfig):
    model_type = "minigram"

    def __init__(
        self,
        dropout: int = 0.1,
        vocab_size: int = 10240,
        num_attention_heads: int = 12,
        num_kv_heads: int = 12,
        num_hidden_layers: int = 12,
        hidden_size: int = 768,
        intermediate_size: int = None,
        hidden_act: str = "gelu",
        initializer_range: int = 0.02,
        use_cache: bool = True,
        max_length: int = 512,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        flash_attention: bool = False,
        rope_scaling_params: dict = None,
        rope_theta: float = 100000.0,
        ##############################################
        # MoE-specific parameters
        # When use_moe is False, the following parameters will be ignored
        ###############################################
        use_moe: bool = False,
        num_experts: int = 4,
        shard_expert: bool = False,
        num_expert_per_token: int = 2,
        scoring_func: str = "softmax",
        aux_loss_coef: float = 0.01,
        ###########################################################
        # engram-specific parameters
        # When use_engrams is False, the following parameters will be ignored
        ###########################################################
        use_engrams: bool = False,
        engram_vocab_size: int = 1024,
        engram_ratio: float = 0.25,
        engram_n_layer_list: list = [2],    # 在第几层使用engrams
        engram_n_gram_list: list = [2, 3],  # 使用2-gram和3-gram
        engram_num_heads: int = 4,          # 每个阶数使用的哈希头数
        engram_conv_size: int = 3,              # 卷积核大小
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout                      # The dropout probability for all fully connected layers in the embeddings, text_encoder, and pooler.
        self.vocab_size = vocab_size                    # The size of the vocabulary, including special tokens.
        self.max_length = max_length                    # The maximum sequence length that the model can process. Sequences longer than this will be truncated.
        self.num_attention_heads = num_attention_heads  # The number of attention heads in each attention layer. This should be a divisor of hidden_size.
        self.num_kv_heads = num_kv_heads                # The number of key-value heads in each attention layer. This should be a divisor of num_attention_heads.
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range      # The standard deviation of the truncated_normal_initializer for initializing all weight matrices. This is used to control the scale of the initial weights, which can affect the convergence of the model during training.
        self.use_cache = use_cache                      # Whether the model should return the key-value states on each forward pass. This is used to speed up decoding by reusing previously computed key-value states.
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.flash_attention = flash_attention
        self.rope_theta = rope_theta                    # The base period of the RoPE. This is used to control the frequency of the sinusoidal functions in the RoPE. A larger theta allows the model to capture longer-range dependencies, while a smaller theta focuses more on local interactions.
        self.rope_factors = {
            "beta_fast": 16.0,   # The scaling factor for the fast attention heads in the RoPE. This is used to control the relative importance of the fast attention heads compared to the slow attention heads in the RoPE. A larger beta_fast gives more weight to the fast attention heads.
            "beta_slow": 1.0,
            "factor": 16,       # The scaling factor for the RoPE. This is used to control the overall scale of the RoPE. A larger factor allows the model to capture longer-range dependencies, while a smaller factor focuses more on local interactions.
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if rope_scaling_params is None else rope_scaling_params
        ###############################
        # MoE-specific parameters
        # When use_moe is False, the following parameters will be ignored
        ###############################
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.shard_expert = shard_expert
        self.num_expert_per_token = num_expert_per_token
        self.scoring_func = scoring_func
        self.aux_loss_coef = aux_loss_coef
        ###########################################################
        # engram-specific parameters
        # When use_engrams is False, the following parameters will be ignored
        ###########################################################
        self.use_engrams = use_engrams
        self.engram_vocab_size = engram_vocab_size
        self.engram_ratio = engram_ratio
        self.engram_n_layer_list = engram_n_layer_list
        self.engram_n_gram_list = engram_n_gram_list
        self.engram_num_heads = engram_num_heads
        self.engram_conv_size = engram_conv_size




from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch import nn
import torch.nn.functional as F

def _precompute_freqs_cis(dim, end, theta=100000.0, params:Optional[dict]=None):
    freqs, attn_factor = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)] / dim).float()), 1.0
    if params is not None:
        beta_fast = params.get("beta_fast", 16.0)
        beta_slow = params.get("beta_slow", 1.0)
        factor = params.get("factor", 16)
        orig_max = params.get("original_max_position_embeddings", 512)
        attn_factor = params.get("attention_factor", 1.0)
        if end > orig_max:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(theta))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) * attn_factor
    freq_cos = torch.cat((freqs_cis.real.float(), freqs_cis.real.float()), dim=-1)
    freq_sin = torch.cat((freqs_cis.imag.float(), freqs_cis.imag.float()), dim=-1)
    return freqs_cis, freq_cos, freq_sin

def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # q,k: (batch, seq, head, dim), cos,sin: (seq, dim)
    cos = cos.unsqueeze(unsqueeze_dim)  # (seq, 1, dim)
    sin = sin.unsqueeze(unsqueeze_dim)  # (seq, 1, dim)
    q_embed = torch.stack([-q[..., 1::2], q[..., ::2]], -1).reshape_as(q)  # (batch, seq, head, dim)
    k_embed = torch.stack([-k[..., 1::2], k[..., ::2]], -1).reshape_as(k)  # (batch, seq, head, dim)
    q_out = q * cos + q_embed * sin
    k_out = k * cos + k_embed * sin
    return q_out, k_out

def repeat_kv(tensor, num_kv_heads, num_attention_heads):
    # tensor: (batch, seq, num_kv_heads, head_dim)
    b, s, _, d = tensor.shape
    if num_kv_heads == num_attention_heads:
        return tensor
    repeat_factor = num_attention_heads // num_kv_heads
    return tensor.unsqueeze(3).repeat_interleave(repeat_factor, dim=3).reshape(b, s, num_attention_heads, d)

def _build_attention_bias(attention_mask, q_len, kv_len, device, dtype, past_length=0):
    causal = torch.triu(
        torch.ones(q_len, kv_len, device=device, dtype=torch.bool),
        diagonal=1 + past_length,
    )
    attn_bias = torch.zeros((1, 1, q_len, kv_len), device=device, dtype=dtype)
    min_value = torch.finfo(dtype).min
    attn_bias = attn_bias.masked_fill(causal.unsqueeze(0).unsqueeze(0), min_value)

    if attention_mask is None:
        return attn_bias

    if attention_mask.dim() == 2:
        key_mask = attention_mask[:, :kv_len].to(device=device)
        key_mask = key_mask[:, None, None, :].to(torch.bool)
        return attn_bias.masked_fill(~key_mask, min_value)

    if attention_mask.dim() == 4:
        return attn_bias + attention_mask.to(device=device, dtype=dtype)

    raise ValueError(f"Unsupported attention_mask shape: {tuple(attention_mask.shape)}")

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class SimpleAttention(nn.Module):
    def __init__(self, config: MiniGramConfig):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_attention_heads = config.num_attention_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, self.head_dim * self.num_kv_heads)
        self.v_proj = nn.Linear(config.hidden_size, self.head_dim * self.num_kv_heads)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout_rate = config.dropout
        self.dropout = nn.Dropout(config.dropout)
        self.rope_theta = config.rope_theta
        self.rope_factors = config.rope_factors
        self.flash_attn = config.flash_attention
    
    def forward(
            self, 
            hidden_states, 
            attention_mask=None, 
            use_cache=False, 
            past_key_value=None,
            precompute_freqs=None
        ):
        batch_size, seq_length, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_kv_heads, self.head_dim)

        if precompute_freqs is not None:
            cos, sin = precompute_freqs
        else:
            _, cos, sin = _precompute_freqs_cis(self.head_dim, seq_length, theta=self.rope_theta, params=self.rope_factors)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        if use_cache:
            if past_key_value is not None:
                k = torch.cat([past_key_value[0], k], dim=1)
                v = torch.cat([past_key_value[1], v], dim=1)
            past_key_value = (k, v)
        else:
            past_key_value = None
            
        if self.num_kv_heads != self.num_attention_heads:
            k = repeat_kv(k, self.num_kv_heads, self.num_attention_heads)
            v = repeat_kv(v, self.num_kv_heads, self.num_attention_heads)


        if self.flash_attn and (seq_length > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout_rate if self.training else 0.0,
                is_causal=True,
            )
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        else:
            kv_length = k.size(1)
            past_length = kv_length - seq_length
            attn_weights = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.head_dim)
            attn_bias = _build_attention_bias(
                attention_mask,
                q_len=seq_length,
                kv_len=kv_length,
                device=hidden_states.device,
                dtype=attn_weights.dtype,
                past_length=past_length,
            )
            attn_weights = attn_weights + attn_bias
            attn_probs = torch.softmax(attn_weights, dim=-1)
            attn_probs = self.dropout(attn_probs)
            attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_probs, v).contiguous().view(batch_size, seq_length, -1)

        output = self.dropout(self.out_proj(attn_output))
        return output, past_key_value


class EngramModule(nn.Module):
    def __init__(self, config: MiniGramConfig):
        super().__init__()
        # 1. 核心超参数
        self.n_gram_list = config.engram_n_gram_list      # e.g., [2, 3]
        self.num_heads = config.engram_num_heads          # K = 8
        self.engram_vocab_size = config.engram_vocab_size # 嵌入表槽数，建议设为质数
        self.hidden_size = config.hidden_size             # d_model = 768
        
        # 2. 嵌入表 (Engram Memory)
        # 为每个 (N-gram阶数, 哈希头) 组合创建一个独立的嵌入表
        self.embeddings = nn.ModuleDict()
        for n in self.n_gram_list:
            for k in range(self.num_heads):
                # 表名如 "emb_2_0", "emb_2_1"...
                self.embeddings[f"emb_{n}_{k}"] = nn.Embedding(
                    self.engram_vocab_size, self.hidden_size
                )
        self.down_proj = nn.Linear(
            self.hidden_size * self.num_heads * len(self.n_gram_list), 
            self.hidden_size
        )
        
        # 3. [门控机制] 动态决定记忆的融合程度
        # 线性层将 hidden_size 映射到 1，用于计算门控标量 alpha_t
        self.gate_proj = nn.Linear(self.hidden_size * 2, 1)
        
        # 4. [后处理卷积] 微调记忆向量（可选）
        self.conv = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=4,
            groups=self.hidden_size,   # 深度卷积
            bias=False
        )
        
        # 5. [输出投影] 将融合后的结果映射回 hidden_size（如果需要）
        # 如果直接相加，可能不需要此层；这里作为一个占位符
        # self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
    
    def compute_n_gram_ids(self, input_ids, n, k):
        """
        Returns:
            n_gram_ids: (batch, seq_len)
            Right-aligned n-gram hash ids, with the first n-1 positions set to 0.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if seq_len == 0:
            return input_ids.new_zeros(batch_size, 0)

        # Per-head mixing coefficients.
        head_bases = [31, 37, 41, 43, 47, 53, 59, 61]
        base = head_bases[k % len(head_bases)]

        if seq_len < n:
            return input_ids.new_zeros(batch_size, seq_len)

        windows = input_ids.unfold(dimension=1, size=n, step=1)  # (batch, seq_len - n + 1, n)

        # Small integer weights to avoid overflow/precision problems.
        weights = input_ids.new_tensor([base + i for i in range(n)])  # (n,)
        hash_vals = (windows * weights).sum(dim=-1) % (self.engram_vocab_size - 1)
        hash_vals = hash_vals + 1  # reserve 0 as the empty / padded slot

        n_gram_ids = input_ids.new_zeros(batch_size, seq_len)
        n_gram_ids[:, n - 1:] = hash_vals
        return n_gram_ids

    def forward(self, hidden_states, input_ids):
        batch_size, seq_len, _ = hidden_states.shape

        # 1. 收集所有头、所有阶数的记忆向量，并左填充到 token 维度对齐
        mem_list = []
        for n in self.n_gram_list:
            for k in range(self.num_heads):
                n_gram_ids = self.compute_n_gram_ids(input_ids, n, k)   # (batch, seq_len)
                emb = self.embeddings[f"emb_{n}_{k}"](n_gram_ids)       # (batch, seq_len, hidden)
                mem_list.append(emb)


        # 2. 拼接并降维到 hidden_size
        concat_mem = torch.cat(mem_list, dim=-1)                        # (batch, seq, hidden * num_heads * len)
        memory_vec = self.down_proj(concat_mem)

        # 3. 计算门控标量（同时依赖 hidden_states 和 memory_vec）
        gate_input = torch.cat([hidden_states, memory_vec], dim=-1)     # (batch, seq, hidden*2)
        alpha = torch.sigmoid(self.gate_proj(gate_input))               # (batch, seq, 1)

        # 4. 残差融合
        fused = hidden_states + alpha * memory_vec                      # (batch, seq, hidden)

        # 5. 因果卷积微调（保持因果性）
        fused = fused.transpose(1, 2)                                   # (batch, hidden, seq)
        fused_pad = F.pad(fused, (3, 0))                               # 因果填充，假设 kernel_size=4
        out = self.conv(fused_pad).transpose(1, 2)                     # (batch, seq, hidden)

        # 6. 输出投影（如果需要）
        # out = self.out_proj(out)

        return out    


class FFN(nn.Module):
    """
    same impl as minimind FeedForward
    """
    
    def __init__(self, config: MiniGramConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))

class TransformerBlock(nn.Module):
    def __init__(self, config: MiniGramConfig):
        super().__init__()
        self.attention = SimpleAttention(config)
        self.ffn = FFN(config)
        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)
    
    def forward(self, hidden_states, attention_mask=None, use_cache=False, past_key_value=None, precompute_freqs=None):
        residual = hidden_states
        attn_output, new_past_key_value = self.attention(
            self.norm1(hidden_states),
            attention_mask,
            use_cache,
            past_key_value,
            precompute_freqs,
        )
        hidden_states = residual + attn_output
        residual = hidden_states
        ffn_output = self.ffn(self.norm2(hidden_states))
        hidden_states = residual + ffn_output
        return hidden_states, new_past_key_value
    

class TransformerBlockWithEngram(nn.Module):
    def __init__(self, config: MiniGramConfig):
        super().__init__()
        self.attention = SimpleAttention(config)
        self.engram = EngramModule(config)
        self.ffn = FFN(config)
        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)
        self.norm3 = RMSNorm(config.hidden_size)

    def forward(self, hidden_states, input_ids, precompute_freqs, attention_mask=None, use_cache=False, past_key_value=None):
        residual = hidden_states
        attn_output, new_past_key_value = self.attention(
            self.norm1(hidden_states),
            attention_mask,
            use_cache,
            past_key_value,
            precompute_freqs
        )
        hidden_states = residual + attn_output

        # Engram模块融合
        engram_output = self.engram(self.norm2(hidden_states), input_ids)
        hidden_states = hidden_states + engram_output


        residual = hidden_states
        ffn_output = self.ffn(self.norm3(hidden_states))
        hidden_states = residual + ffn_output
        return hidden_states, new_past_key_value


class MiniGramModel(nn.Module):
    def __init__(self, config: MiniGramConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            if config.use_engrams and i in config.engram_n_layer_list:
                self.layers.append(TransformerBlockWithEngram(config))
            else:
                self.layers.append(TransformerBlock(config))
        self.norm = RMSNorm(config.hidden_size)
        _, cos, sin = _precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads,
            config.max_length, 
            theta=config.rope_theta, 
            params=config.rope_factors
        )
        self.register_buffer("precompute_freqs_cos", cos)
        self.register_buffer("precompute_freqs_sin", sin)
    
    def forward(self, input_ids, attention_mask=None, use_cache=False, past_key_values=None):
        hidden_states = self.token_embedding(input_ids)
        new_past_key_values = []
        seq_length = input_ids.size(1)
        past_length = past_key_values[0][0].size(1) if past_key_values is not None else 0
        precompute_freqs = (
            self.precompute_freqs_cos[past_length:past_length + seq_length],
            self.precompute_freqs_sin[past_length:past_length + seq_length],
        )
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if isinstance(layer, TransformerBlockWithEngram):
                hidden_states, new_past_key_value = layer(
                    hidden_states, input_ids, precompute_freqs, attention_mask, use_cache, past_key_value
                )
            else:
                hidden_states, new_past_key_value = layer(
                    hidden_states, attention_mask, use_cache, past_key_value, precompute_freqs
                )
            new_past_key_values.append(new_past_key_value)
        hidden_states = self.norm(hidden_states)
        return hidden_states, new_past_key_values


class MiniGramForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniGramConfig

    def __init__(self, config: MiniGramConfig):
        super().__init__(config)
        self.model = MiniGramModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.model.token_embedding.weight = self.lm_head.weight
        self.use_cache = config.use_cache
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        use_cache=None,
        past_key_values=None,
        labels=None,
        logits_to_keep=0,
        **kwargs
    ):
        use_cache = self.use_cache if use_cache is None else use_cache

        hidden_states, new_past_key_values = self.model(
            input_ids, attention_mask, use_cache, past_key_values
        )
        logits_hidden_states = hidden_states
        if labels is None and logits_to_keep != 0:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits_hidden_states = hidden_states[:, slice_indices, :]

        logits = self.lm_head(logits_hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_past_key_values,
        )
        setattr(output, "aux_loss", logits.new_zeros(()))
        return output
