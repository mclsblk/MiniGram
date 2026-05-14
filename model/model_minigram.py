from transformers import PretrainedConfig
from typing import Optional
import torch
import math

class MiniGramConfig(PretrainedConfig):
    model_type = "minigram"

    def __init__(self, hidden_size: int = 768, num_hidden_layers: int = 12, **kwargs):
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        
        self.dropout = kwargs.get("dropout", 0.1)
        self.vocab_size = kwargs.get("vocab_size", 10240)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_kv_heads = kwargs.get("num_kv_heads", 4)
        self.intermediate_size = kwargs.get("intermediate_size", None)
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.initializer_range = kwargs.get("initializer_range", 0.02)
        self.use_cache = kwargs.get("use_cache", True)
        self.max_length = kwargs.get("max_length", 32768)
        self.bos_token_id = kwargs.get("bos_token_id", 0)
        self.eos_token_id = kwargs.get("eos_token_id", 1)
        self.flash_attention = kwargs.get("flash_attention", True if torch.cuda.is_available() else False)
        rope_scaling_params = kwargs.get("rope_scaling_params", None)
        self.rope_theta = kwargs.get("rope_theta", 100000.0)
        
        # MoE parameters
        self.use_moe = kwargs.get("use_moe", False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_expert_per_token = kwargs.get("num_expert_per_token", 2)
        self.aux_loss_coef = kwargs.get("aux_loss_coef", 0.01)
        
        # Engram parameters
        self.use_engrams = kwargs.get("use_engrams", False)
        self.engram_vocab_size = kwargs.get("engram_vocab_size", 1024)
        self.engram_n_layer_list = kwargs.get("engram_n_layer_list", [1])
        self.engram_n_gram_list = kwargs.get("engram_n_gram_list", [2, 3])
        self.engram_num_heads = kwargs.get("engram_num_heads", 4)
        self.engram_conv_size = kwargs.get("engram_conv_size", 3)
        self.engram_hash_seed = kwargs.get("engram_hash_seed", 17)
        
        self.rope_factors = {
            "beta_fast": 16.0,
            "beta_slow": 1.0,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if rope_scaling_params is None else rope_scaling_params


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
    freq_cos = freqs_cis.real.float().repeat_interleave(2, dim=-1)
    freq_sin = freqs_cis.imag.float().repeat_interleave(2, dim=-1)
    return freqs_cis, freq_cos, freq_sin

def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # q,k: (batch, seq, head, dim), cos,sin: (seq, dim)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch.stack([-q[..., 1::2], q[..., ::2]], -1).reshape_as(q)
    k_embed = torch.stack([-k[..., 1::2], k[..., ::2]], -1).reshape_as(k)
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

def _get_from_cache(past_key_value, key):
    if past_key_value is None:
        return None
    if isinstance(past_key_value, dict):
        return past_key_value.get(key)
    return None

def _get_past_length(past_key_value):
    if past_key_value is None:
        return 0
    if isinstance(past_key_value, dict) and "attn" in past_key_value:
        attn_cache = past_key_value["attn"]
        if attn_cache is not None and isinstance(attn_cache, tuple) and len(attn_cache) == 2:
            k_cache = attn_cache[0]
            if k_cache is not None and k_cache.dim() >= 2:
                return k_cache.size(1)
    return 0

def _normalize_past_key_values(past_key_values):
    if past_key_values is None:
        return None

    to_legacy = getattr(past_key_values, "to_legacy_cache", None)
    if callable(to_legacy):
        legacy_cache = to_legacy()
        if legacy_cache is not None:
            return legacy_cache

    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        return layers

    return past_key_values


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
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.head_dim * self.num_kv_heads, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.head_dim * self.num_kv_heads, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout_rate = config.dropout
        self.dropout = nn.Dropout(config.dropout)
        self.rope_theta = config.rope_theta
        self.rope_factors = config.rope_factors
        self.flash_attn = config.flash_attention
    
    def forward(self, hidden_states, precompute_freqs, 
                attention_mask=None, use_cache=False, past_key_value=None):
        batch_size, seq_length, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_kv_heads, self.head_dim)

        cos, sin = precompute_freqs
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
            attn_output = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=self.dropout_rate if self.training else 0.0,
                is_causal=True,
            )
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        else:
            kv_length = k.size(1)
            past_length = kv_length - seq_length
            attn_weights = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.head_dim)
            min_value = -1e9 if attn_weights.dtype == torch.float32 else -1e4
            causal_mask = torch.triu(
                torch.ones(seq_length, kv_length, device=hidden_states.device, dtype=torch.bool),
                diagonal=1 + past_length,
            )
            attn_bias = torch.zeros((1, 1, seq_length, kv_length), device=hidden_states.device, dtype=attn_weights.dtype)
            attn_bias = attn_bias.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), min_value)

            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    key_mask = attention_mask[:, :kv_length].to(device=hidden_states.device, dtype=torch.bool)
                    attn_bias = attn_bias.masked_fill(~key_mask[:, None, None, :], min_value)
                elif attention_mask.dim() == 4:
                    attn_bias = attn_bias + attention_mask.to(device=hidden_states.device, dtype=attn_weights.dtype)
                else:
                    raise ValueError(f"Unsupported attention_mask shape: {tuple(attention_mask.shape)}")

            attn_weights = attn_weights + attn_bias
            attn_probs = torch.softmax(attn_weights, dim=-1)
            attn_probs = self.dropout(attn_probs)
            attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_probs, v).contiguous().view(batch_size, seq_length, -1)

        output = self.dropout(self.out_proj(attn_output))
        return output, past_key_value


class EngramModule(nn.Module):
    def __init__(self, config: MiniGramConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.n_gram_list = config.engram_n_gram_list
        self.num_heads = config.engram_num_heads
        self.engram_vocab_size = config.engram_vocab_size
        self.hidden_size = config.hidden_size
        self.hash_seed = config.engram_hash_seed
        self.conv_kernel_size = config.engram_conv_size
        self.total_memory_heads = len(self.n_gram_list) * self.num_heads
        self.head_dim = max(1, math.ceil(self.hidden_size / self.total_memory_heads))
        self.memory_dim = self.head_dim * self.total_memory_heads
        self.hash_modulus = self.engram_vocab_size - 1
        self.token_tail_size = max(self.n_gram_list) - 1
        self.conv_tail_size = self.conv_kernel_size - 1

        self.head_slices = {}
        head_offset = 0
        for n in self.n_gram_list:
            self.head_slices[n] = slice(head_offset, head_offset + self.num_heads)
            head_offset += self.num_heads

        self.hash_multiplier_names = {}
        self.hash_offset_names = {}
        for n in self.n_gram_list:
            multipliers, offsets = self._build_hash_parameters(n)
            multiplier_name = f"hash_multipliers_{n}"
            offset_name = f"hash_offsets_{n}"
            self.register_buffer(multiplier_name, multipliers, persistent=False)
            self.register_buffer(offset_name, offsets, persistent=False)
            self.hash_multiplier_names[n] = multiplier_name
            self.hash_offset_names[n] = offset_name

        self.embeddings = nn.ModuleList(
            [nn.Embedding(self.engram_vocab_size, self.head_dim, padding_idx=0) for _ in range(self.total_memory_heads)]
        )
        self.memory_key_proj = nn.Linear(self.memory_dim, self.hidden_size, bias=False)
        self.memory_value_proj = nn.Linear(self.memory_dim, self.hidden_size, bias=False)
        self.memory_key_norm = RMSNorm(self.hidden_size)
        self.memory_value_norm = RMSNorm(self.hidden_size)
        self.memory_conv = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=self.conv_kernel_size,
            groups=self.hidden_size,
            bias=False,
        )
        self.memory_gate_bias = nn.Parameter(torch.tensor(-4.0))
        nn.init.zeros_(self.memory_conv.weight)

    def _build_hash_parameters(self, n: int):
        multipliers = []
        offsets = []
        max_int = (1 << 31) - 1
        for head_idx in range(self.num_heads):
            base_seed = (
                self.hash_seed
                + 10007 * (self.layer_id + 1)
                + 1543 * (n + 1)
                + 8191 * (head_idx + 1)
            )
            head_multipliers = []
            for pos in range(n):
                value = (base_seed + 32771 * (pos + 1) + 65537 * (head_idx + 1) * (pos + 1)) % max_int
                head_multipliers.append(value * 2 + 1)
            offset = (base_seed * 48271 + 97 * (n + head_idx + 1)) % max_int
            multipliers.append(head_multipliers)
            offsets.append(offset)
        return torch.tensor(multipliers, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)

    def compute_hash_ids(self, input_ids, tail_tokens=None):
        batch_size, seq_len = input_ids.shape
        if tail_tokens is None:
            tail_tokens = input_ids.new_zeros(batch_size, 0)
        else:
            tail_tokens = tail_tokens.to(device=input_ids.device, dtype=input_ids.dtype)

        if seq_len == 0:
            empty_hashes = input_ids.new_zeros(batch_size, 0, self.total_memory_heads)
            new_tail = tail_tokens[:, -self.token_tail_size:] if self.token_tail_size > 0 else input_ids.new_zeros(batch_size, 0)
            return empty_hashes, new_tail

        context = torch.cat([tail_tokens, input_ids], dim=1)
        prefix_len = tail_tokens.size(1)
        hash_ids = input_ids.new_zeros(batch_size, seq_len, self.total_memory_heads)

        if self.hash_modulus <= 0:
            new_tail = context[:, -self.token_tail_size:] if self.token_tail_size > 0 else input_ids.new_zeros(batch_size, 0)
            return hash_ids, new_tail

        for n in self.n_gram_list:
            full_hash = input_ids.new_zeros(batch_size, context.size(1), self.num_heads)
            if context.size(1) >= n:
                windows = context.unfold(dimension=1, size=n, step=1).to(torch.long)
                multipliers = getattr(self, self.hash_multiplier_names[n]).to(device=input_ids.device)
                offsets = getattr(self, self.hash_offset_names[n]).to(device=input_ids.device)
                mix = windows[:, :, 0].unsqueeze(-1) * multipliers[:, 0].view(1, 1, -1)
                for pos in range(1, n):
                    current = windows[:, :, pos].unsqueeze(-1) * multipliers[:, pos].view(1, 1, -1)
                    mix = torch.bitwise_xor(mix, current)
                full_hash[:, n - 1:, :] = torch.remainder(mix + offsets.view(1, 1, -1), self.hash_modulus) + 1
            hash_ids[:, :, self.head_slices[n]] = full_hash[:, prefix_len:prefix_len + seq_len, :]

        if self.token_tail_size > 0:
            new_tail = context[:, -self.token_tail_size:]
        else:
            new_tail = input_ids.new_zeros(batch_size, 0)
        return hash_ids, new_tail

    def retrieve_memory(self, hash_ids):
        head_embeddings = [embedding(hash_ids[:, :, head_idx]) for head_idx, embedding in enumerate(self.embeddings)]
        return torch.cat(head_embeddings, dim=-1)

    def apply_memory_conv(self, memory_value, conv_state=None):
        batch_size, seq_len, _ = memory_value.shape
        if conv_state is None:
            conv_state = memory_value.new_zeros(batch_size, 0, self.hidden_size)
        else:
            conv_state = conv_state.to(device=memory_value.device, dtype=memory_value.dtype)

        conv_source = torch.cat([conv_state, memory_value], dim=1)
        conv_full = self.memory_conv(F.pad(conv_source.transpose(1, 2), (self.conv_tail_size, 0))).transpose(1, 2)
        conv_current = conv_full[:, -seq_len:, :]
        if self.conv_tail_size > 0:
            new_conv_state = conv_source[:, -self.conv_tail_size:, :]
        else:
            new_conv_state = memory_value.new_zeros(batch_size, 0, self.hidden_size)
        return conv_current, new_conv_state

    def forward(self, hidden_states, input_ids, tail_tokens=None, conv_state=None):
        hash_ids, new_tail = self.compute_hash_ids(input_ids, tail_tokens=tail_tokens)
        memory = self.retrieve_memory(hash_ids)
        memory_key = self.memory_key_norm(self.memory_key_proj(memory))
        memory_value = self.memory_value_norm(self.memory_value_proj(memory))
        gate_logits = (hidden_states * memory_key).sum(dim=-1) / math.sqrt(self.hidden_size)
        gate_logits = gate_logits + self.memory_gate_bias
        gate = torch.sigmoid(gate_logits).unsqueeze(-1)
        gated_memory = gate * memory_value
        conv_out, new_conv_state = self.apply_memory_conv(gated_memory, conv_state=conv_state)
        return gated_memory + conv_out, new_tail, new_conv_state

# ############################################################################ #
# FFN and FFNofMoE are same implementation from minimind                       #
# ############################################################################ #
class FFN(nn.Module): 
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

class FFNofMoE(nn.Module):
    def __init__(self, config: MiniGramConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [FFN(config) for _ in range(config.num_experts)]
        )
        self.act_fn = ACT2FN[config.hidden_act]
        self.loss_coef = config.aux_loss_coef
        
    def forward(self, x: torch.Tensor):
        batch_size, seq_length, hidden_size = x.size()
        x_flat = x.view(-1, hidden_size)
        gate_probs = F.softmax(self.gate(x_flat), dim=-1)
        topk_probs, topk_indices = torch.topk(gate_probs, self.config.num_expert_per_token, dim=-1)
        expert_outputs = torch.zeros_like(x_flat)
        for indices, expert in enumerate(self.experts):
            mask = (topk_indices == indices).any(dim=-1)
            if mask.any():
                weight = topk_probs[mask].view(-1, 1)
                expert_output = expert(x_flat[mask]).view(-1, 1) * weight
                expert_outputs.index_add_(0, mask.any(dim=-1).nonzero().flatten(), expert_output)
            elif self.training:
                expert_outputs = expert_outputs + 0.0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.loss_coef > 0:
            load = F.one_hot(topk_indices, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * gate_probs.mean(0)).sum() * self.config.num_experts * self.loss_coef
        else:
            self.aux_loss = torch.tensor(0.0, device=x.device)
        return expert_outputs.view(batch_size, seq_length, hidden_size)

class TransformerBlock(nn.Module):
    def __init__(self, config: MiniGramConfig, layer_id: int = None):
        super().__init__()
        self.attention = SimpleAttention(config)
        self.ffn = FFN(config) if not config.use_moe else FFNofMoE(config)   
        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)
        if config.use_engrams and (layer_id is not None) and (layer_id in config.engram_n_layer_list):
            self.layer_id = layer_id
            self.engram = EngramModule(config, layer_id=layer_id)
            self.norm_engram = RMSNorm(config.hidden_size)
        else:
            self.engram = None
    
    def forward(self, hidden_states, attention_mask=None, use_cache=False, 
                past_key_value=None, precompute_freqs=None, input_ids=None):
        residual = hidden_states
        attn_output, new_past_key_value = self.attention(
            self.norm1(hidden_states),
            precompute_freqs,
            attention_mask,
            use_cache,
            _get_from_cache(past_key_value, "attn"),
        )
        hidden_states = residual + attn_output
        layer_cache = {"attn": new_past_key_value} if use_cache else None
        
        if self.engram is not None:
            engram_output, new_engram_tail, new_engram_conv = self.engram(
            self.norm_engram(hidden_states),
            input_ids,
            tail_tokens=_get_from_cache(past_key_value, "engram_tail"),
            conv_state=_get_from_cache(past_key_value, "engram_conv")
            )
            hidden_states = hidden_states + engram_output
            if use_cache:
                layer_cache["engram_tail"] = new_engram_tail
                layer_cache["engram_conv"] = new_engram_conv

        residual = hidden_states
        ffn_output = self.ffn(self.norm2(hidden_states))
        hidden_states = residual + ffn_output
        return hidden_states, layer_cache


class MiniGramModel(nn.Module):
    def __init__(self, config: MiniGramConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            self.layers.append(TransformerBlock(config, layer_id=i))
        self.norm = RMSNorm(config.hidden_size)
        _, cos, sin = _precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads,
            config.max_length, 
            theta=config.rope_theta, 
            params=config.rope_factors
        )
        self.register_buffer("precompute_freqs_cos", cos, persistent=False)
        self.register_buffer("precompute_freqs_sin", sin, persistent=False)
    
    def forward(self, input_ids, attention_mask=None, use_cache=False, past_key_values=None):
        hidden_states = self.token_embedding(input_ids)
        new_past_key_values = []
        seq_length = input_ids.size(1)
        past_key_values = _normalize_past_key_values(past_key_values)
        past_key_values = past_key_values or [None] * len(self.layers)
        past_length = _get_past_length(past_key_values[0]) if past_key_values else 0
        precompute_freqs = (
            self.precompute_freqs_cos[past_length:past_length + seq_length],
            self.precompute_freqs_sin[past_length:past_length + seq_length],
        )
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            hidden_states, new_past_key_value = layer(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_value=past_key_value,
                precompute_freqs=precompute_freqs,
                input_ids=input_ids,
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
    
    def forward(self, input_ids=None, attention_mask=None, use_cache=None,
                past_key_values=None, labels=None, logits_to_keep=0, **kwargs):
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


    def _reorder_cache(self, past_key_values, beam_idx):
        if past_key_values is None:
            return past_key_values

        reordered = []
        for layer_cache in past_key_values:
            if layer_cache is None:
                reordered.append(None)
                continue

            if isinstance(layer_cache, dict):
                reordered_layer_cache = {}
                for key, value in layer_cache.items():
                    if value is None:
                        reordered_layer_cache[key] = None
                    elif isinstance(value, tuple):
                        reordered_layer_cache[key] = tuple(
                            tensor.index_select(0, beam_idx.to(tensor.device)) for tensor in value
                        )
                    else:
                        reordered_layer_cache[key] = value.index_select(0, beam_idx.to(value.device))
                reordered.append(reordered_layer_cache)
            else:
                reordered.append(
                    tuple(tensor.index_select(0, beam_idx.to(tensor.device)) for tensor in layer_cache)
                )
        return reordered