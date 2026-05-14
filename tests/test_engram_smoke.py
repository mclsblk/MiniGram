import torch

from model.model_minigram import EngramModule, MiniGramConfig, MiniGramForCausalLM


def make_config(**overrides):
    config = MiniGramConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        max_length=32,
        dropout=0.0,
        flash_attention=False,
        use_cache=True,
        use_engrams=True,
        engram_vocab_size=31,
        engram_ratio=0.5,
        engram_n_layer_list=[0],
        engram_n_gram_list=[2, 3],
        engram_num_heads=2,
        engram_conv_size=3,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_forward_shapes_without_engrams():
    torch.manual_seed(0)
    config = make_config(use_engrams=False, use_cache=False)
    model = MiniGramForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 6))

    outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)

    assert outputs.logits.shape == (2, 6, config.vocab_size)
    assert outputs.loss.ndim == 0
    assert len(outputs.past_key_values) == config.num_hidden_layers


def test_forward_shapes_with_engrams():
    torch.manual_seed(0)
    config = make_config(engram_n_layer_list=[0, 1])
    model = MiniGramForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 7))

    outputs = model(input_ids=input_ids, use_cache=True)

    assert outputs.logits.shape == (2, 7, config.vocab_size)
    assert len(outputs.past_key_values) == config.num_hidden_layers
    assert isinstance(outputs.past_key_values[0], dict)
    assert outputs.past_key_values[0]["engram_tail"].shape == (2, max(config.engram_n_gram_list) - 1)


def test_hash_padding_keeps_first_positions_empty():
    config = make_config()
    module = EngramModule(config, layer_id=0)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    hash_ids, _ = module.compute_hash_ids(input_ids)

    for n in module.n_gram_list:
        head_slice = module.head_slices[n]
        assert torch.count_nonzero(hash_ids[:, : n - 1, head_slice]) == 0
        assert torch.count_nonzero(hash_ids[:, n - 1 :, head_slice]) > 0


def test_hashes_change_across_layers():
    config = make_config()
    input_ids = torch.tensor([[4, 8, 15, 16, 23, 42]])
    module_layer_0 = EngramModule(config, layer_id=0)
    module_layer_1 = EngramModule(config, layer_id=1)

    hash_ids_0, _ = module_layer_0.compute_hash_ids(input_ids)
    hash_ids_1, _ = module_layer_1.compute_hash_ids(input_ids)

    assert not torch.equal(hash_ids_0, hash_ids_1)


def test_memory_only_conv_is_causal_and_shape_preserving():
    torch.manual_seed(0)
    config = make_config()
    module = EngramModule(config, layer_id=0)
    memory_value = torch.randn(1, 5, config.hidden_size)
    memory_value_with_future_change = memory_value.clone()
    memory_value_with_future_change[:, -1, :] += 1.0

    conv_out_a, _ = module.apply_memory_conv(memory_value)
    conv_out_b, _ = module.apply_memory_conv(memory_value_with_future_change)

    assert conv_out_a.shape == memory_value.shape
    assert torch.allclose(conv_out_a[:, :-1, :], conv_out_b[:, :-1, :], atol=1e-6, rtol=1e-6)


def test_cached_engram_matches_full_sequence():
    torch.manual_seed(0)
    config = make_config()
    module = EngramModule(config, layer_id=0).eval()
    hidden_states = torch.randn(1, 6, config.hidden_size)
    input_ids = torch.randint(0, config.vocab_size, (1, 6))

    full_output, _, _ = module(hidden_states, input_ids)

    cached_tail = None
    cached_conv = None
    cached_outputs = []
    for idx in range(input_ids.size(1)):
        step_output, cached_tail, cached_conv = module(
            hidden_states[:, idx : idx + 1, :],
            input_ids[:, idx : idx + 1],
            tail_tokens=cached_tail,
            conv_state=cached_conv,
        )
        cached_outputs.append(step_output)
    cached_output = torch.cat(cached_outputs, dim=1)

    assert torch.allclose(cached_output, full_output, atol=1e-5, rtol=1e-5)


def test_model_cached_forward_matches_full_sequence():
    torch.manual_seed(0)
    config = make_config(engram_n_layer_list=[0, 1], use_cache=True)
    model = MiniGramForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 6))

    full_logits = model(input_ids=input_ids, use_cache=False).logits

    past_key_values = None
    cached_logits = []
    for idx in range(input_ids.size(1)):
        outputs = model(
            input_ids=input_ids[:, idx : idx + 1],
            use_cache=True,
            past_key_values=past_key_values,
        )
        cached_logits.append(outputs.logits)
        past_key_values = outputs.past_key_values

    cached_logits = torch.cat(cached_logits, dim=1)
    assert torch.allclose(cached_logits, full_logits, atol=1e-5, rtol=1e-5)


def test_model_accepts_cache_wrapped_in_layers_attr():
    class FakeCache:
        def __init__(self, layers):
            self.layers = layers

    torch.manual_seed(0)
    config = make_config(engram_n_layer_list=[0, 1], use_cache=True)
    model = MiniGramForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 4))

    first = model(input_ids=input_ids[:, :2], use_cache=True)
    wrapped_cache = FakeCache(first.past_key_values)

    wrapped = model(
        input_ids=input_ids[:, 2:3],
        use_cache=True,
        past_key_values=wrapped_cache,
    )
    unwrapped = model(
        input_ids=input_ids[:, 2:3],
        use_cache=True,
        past_key_values=first.past_key_values,
    )

    assert torch.allclose(wrapped.logits, unwrapped.logits, atol=1e-5, rtol=1e-5)
    assert len(wrapped.past_key_values) == len(unwrapped.past_key_values)


def test_engram_parameters_scale_with_vocab_and_ratio():
    small_config = make_config(engram_vocab_size=17, engram_ratio=0.25)
    large_config = make_config(engram_vocab_size=61, engram_ratio=0.75)
    small_model = MiniGramForCausalLM(small_config)
    large_model = MiniGramForCausalLM(large_config)

    small_params = sum(parameter.numel() for parameter in small_model.parameters())
    large_params = sum(parameter.numel() for parameter in large_model.parameters())

    assert large_params > small_params
