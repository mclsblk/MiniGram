import torch

from model.lora import (
    LoRALinear,
    apply_lora,
    load_lora_state_dict,
    lora_state_dict,
    mark_only_lora_as_trainable,
    merge_lora_weights,
)
from model.model_minigram import MiniGramConfig, MiniGramForCausalLM


def make_model(seed=0):
    torch.manual_seed(seed)
    config = MiniGramConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        max_length=32,
        dropout=0.0,
        flash_attention=False,
        use_cache=False,
        use_engrams=False,
    )
    return MiniGramForCausalLM(config).eval()


def randomize_lora(model):
    torch.manual_seed(123)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.weight.data.normal_(0.0, 0.02)
            module.lora_B.weight.data.normal_(0.0, 0.02)


def test_lora_initial_output_is_unchanged():
    model = make_model()
    input_ids = torch.randint(0, 64, (2, 6))
    with torch.no_grad():
        before = model(input_ids=input_ids).logits

    replaced = apply_lora(model, ["q_proj", "v_proj"], r=4, alpha=8, dropout=0.0)
    with torch.no_grad():
        after = model(input_ids=input_ids).logits

    assert replaced == 4
    assert torch.allclose(before, after, atol=1e-6, rtol=1e-6)


def test_only_lora_parameters_are_trainable():
    model = make_model()
    apply_lora(model, ["q_proj", "v_proj"], r=4, alpha=8, dropout=0.0)
    mark_only_lora_as_trainable(model)

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(".lora_A." in name or ".lora_B." in name for name in trainable)


def test_lora_state_dict_round_trip():
    input_ids = torch.randint(0, 64, (2, 6))
    model_a = make_model(seed=7)
    apply_lora(model_a, ["q_proj", "v_proj"], r=4, alpha=8, dropout=0.0)
    randomize_lora(model_a)

    model_b = make_model(seed=7)
    apply_lora(model_b, ["q_proj", "v_proj"], r=4, alpha=8, dropout=0.0)
    load_lora_state_dict(model_b, lora_state_dict(model_a))

    with torch.no_grad():
        assert torch.allclose(model_a(input_ids=input_ids).logits, model_b(input_ids=input_ids).logits, atol=1e-6)


def test_merge_preserves_output_and_removes_wrappers():
    input_ids = torch.randint(0, 64, (2, 6))
    model = make_model(seed=9)
    apply_lora(model, ["q_proj", "v_proj"], r=4, alpha=8, dropout=0.0)
    randomize_lora(model)
    with torch.no_grad():
        before = model(input_ids=input_ids).logits

    assert merge_lora_weights(model) == 4
    with torch.no_grad():
        after = model(input_ids=input_ids).logits

    assert not any(isinstance(module, LoRALinear) for module in model.modules())
    assert torch.allclose(before, after, atol=1e-5, rtol=1e-5)
