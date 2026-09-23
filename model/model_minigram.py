from dataclasses import dataclass
from transformers import PretrainedConfig
from typing import Optional
import torch
import math

from .channels import build_residual_channel
from .engram import build_engram_layers, resolve_engram_spec
from .validation import (
    validate_model_config, validate_engram_combination, validate_past_key_values,
)


class MiniGramConfig(PretrainedConfig):
    model_type = "minigram"

    def __init__(
        self, hidden_size: int = 768, num_hidden_layers: int = 12,
        use_engrams: bool = False, engram_variant: str = "deepseek",
        engram_overrides: Optional[dict] = None,
        engram_n_layer_list: Optional[list[int]] = None,
        residual_variant: str = "single", residual_low_rank: Optional[int] = None,
        **kwargs,
    ):
        layers = [1] if engram_n_layer_list is None else engram_n_layer_list
        rank = min(64, hidden_size) if residual_low_rank is None else residual_low_rank
        validate_model_config(
            hidden_size, num_hidden_layers, residual_variant, rank,
            layers, use_engrams, kwargs,
        )
        channels = {"single": 1, "gr4": 4, "mhc4": 4}[residual_variant]
        kwargs.pop("residual_channels", None)
        spec = resolve_engram_spec(engram_variant, engram_overrides, hidden_size)
        validate_engram_combination(
            use_engrams, engram_variant, residual_variant, spec.readout, spec.postprocessor,
        )
        # to_dict() stores rope_factors; preserve it when reconstructing a config.
        saved_rope_factors = kwargs.pop("rope_factors", None)
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
        rope_scaling_params = kwargs.get("rope_scaling_params", saved_rope_factors)
        self.rope_theta = kwargs.get("rope_theta", 100000.0)
        
        # MoE parameters
        self.use_moe = kwargs.get("use_moe", False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_expert_per_token = kwargs.get("num_expert_per_token", 2)
        self.aux_loss_coef = kwargs.get("aux_loss_coef", 0.01)
        
        # Only resolved, JSON-compatible configuration is retained, never modules.
        self.use_engrams = use_engrams
        self.engram_variant = engram_variant
        self.engram_overrides = spec.to_dict()
        self.engram_n_layer_list = sorted(layers)
        self.residual_variant = residual_variant
        self.residual_channels = channels
        self.residual_low_rank = rank
        
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


@dataclass
class MiniGramCausalLMOutputWithPast(CausalLMOutputWithPast):
    aux_loss: Optional[torch.FloatTensor] = None

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
    return None if past_key_value is None else past_key_value.get(key)

def _get_past_length(past_key_value):
    if past_key_value is None:
        return 0
    attn_cache = past_key_value["attn"]
    return 0 if attn_cache is None else attn_cache[0].size(1)



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
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        expert_outputs = torch.zeros_like(x_flat)
        for expert_idx, expert in enumerate(self.experts):
            token_indices, topk_slots = torch.where(topk_indices == expert_idx)
            if token_indices.numel() > 0:
                weight = topk_probs[token_indices, topk_slots].unsqueeze(-1)
                expert_output = expert(x_flat[token_indices]) * weight
                expert_outputs.index_add_(0, token_indices, expert_output)
            elif self.training:
                expert_outputs = expert_outputs + 0.0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.loss_coef > 0:
            load = F.one_hot(topk_indices, self.config.num_experts).float().mean(dim=(0, 1))
            aux_loss = (load * gate_probs.mean(0)).sum() * self.config.num_experts * self.loss_coef
        else:
            aux_loss = x.new_zeros(())
        return expert_outputs.view(batch_size, seq_length, hidden_size), aux_loss

class TransformerBlock(nn.Module):
    def __init__(self, config: MiniGramConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.attention = SimpleAttention(config)
        self.use_moe = config.use_moe
        self.ffn = FFN(config) if not config.use_moe else FFNofMoE(config)

    def forward(self, state, channel, engram=None, attention_mask=None, use_cache=False,
                past_key_value=None, precompute_freqs=None, input_ids=None):
        engram_state = _get_from_cache(past_key_value, "engram")
        token_mask = None
        if attention_mask is not None and attention_mask.dim() == 2:
            token_mask = attention_mask[:, -input_ids.size(1):]
        if engram is not None and engram.before_attention:
            delta, engram_state = engram(input_ids, state.streams, engram_state, token_mask)
            state = channel.inject(state, delta)

        hidden, context = channel.read(state, (self.layer_id, "attention"))
        attn_output, attn_cache = self.attention(
            hidden, precompute_freqs, attention_mask, use_cache,
            _get_from_cache(past_key_value, "attn"),
        )
        state = channel.write(state, attn_output, context)

        if engram is not None and not engram.before_attention:
            delta, engram_state = engram(input_ids, state.streams, engram_state, token_mask)
            state = channel.inject(state, delta)

        hidden, context = channel.read(state, (self.layer_id, "ffn"))
        if self.use_moe:
            ffn_output, aux_loss = self.ffn(hidden)
        else:
            ffn_output = self.ffn(hidden)
            aux_loss = hidden.new_zeros(())
        state = channel.write(state, ffn_output, context)
        layer_cache = None
        if use_cache:
            layer_cache = {"attn": attn_cache}
            if engram is not None:
                layer_cache["engram"] = engram_state
        return state, layer_cache, aux_loss


class MiniGramModel(nn.Module):
    def __init__(self, config: MiniGramConfig):
        super().__init__()
        self.config = config
        self.channel = build_residual_channel(config)
        self.engrams = build_engram_layers(config)
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            self.layers.append(TransformerBlock(config, layer_id=i))
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
        state = self.channel.initialize(hidden_states)
        new_past_key_values = [] if use_cache else None
        aux_loss = hidden_states.new_zeros(())
        seq_length = input_ids.size(1)
        validate_past_key_values(past_key_values, len(self.layers), use_cache)
        past_length = _get_past_length(past_key_values[0]) if past_key_values else 0
        precompute_freqs = (
            self.precompute_freqs_cos[past_length:past_length + seq_length],
            self.precompute_freqs_sin[past_length:past_length + seq_length],
        )
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            engram = self.engrams[str(i)] if str(i) in self.engrams else None
            state, new_past_key_value, layer_aux_loss = layer(
                state, channel=self.channel, engram=engram,
                attention_mask=attention_mask, use_cache=use_cache,
                past_key_value=past_key_value, precompute_freqs=precompute_freqs,
                input_ids=input_ids,
            )
            aux_loss = aux_loss + layer_aux_loss
            if use_cache:
                new_past_key_values.append(new_past_key_value)
        return self.channel.finalize(state), new_past_key_values, aux_loss


class MiniGramForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniGramConfig

    def __init__(self, config: MiniGramConfig):
        super().__init__(config)
        self.model = MiniGramModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.model.token_embedding.weight = self.lm_head.weight
        self.use_cache = config.use_cache
        self.post_init()
        self.model.channel.reset_special_parameters()
        for engram in self.model.engrams.values():
            engram.reset_special_parameters()
    
    def forward(self, input_ids=None, attention_mask=None, use_cache=None,
                past_key_values=None, labels=None, logits_to_keep=0, **kwargs):
        use_cache = self.use_cache if use_cache is None else use_cache

        hidden_states, new_past_key_values, aux_loss = self.model(
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

        output = MiniGramCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_past_key_values,
            aux_loss=aux_loss,
        )
        return output


    def _reorder_cache(self, past_key_values, beam_idx):
        validate_past_key_values(past_key_values, self.config.num_hidden_layers)
        caches = past_key_values
        if caches is None:
            return None
        reordered = []
        for cache in caches:
            attn = cache.get("attn")
            layer_cache = {
                "attn": None if attn is None else tuple(
                    value.index_select(0, beam_idx.to(value.device)) for value in attn
                )
            }
            if "engram" in cache:
                state = cache["engram"]
                layer_cache["engram"] = None if state is None else state.reorder(beam_idx)
            reordered.append(layer_cache)
        return reordered
