from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.bounded_att import WindowCompressorState, WindowCompressorStateConfig
from lightllm.common.attention_state.full_att.packed_mla.fp8_e8m0_packed_mla_state import (
    FP8E8M0PackedMLAConfig,
    FP8E8M0PackedMLAState,
)
from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward
from lightllm.models.glm5_next.indexer import Glm5NextNsaInfer
from lightllm.models.glm5_next.layer_infer.transformer_layer_infer import Glm5NextTransformerLayerInfer
from lightllm.models.glm5_next_mtp.model import Glm5NextMTPModel
from lightllm.server.core.objs.start_args_type import StartArgs


def make_model():
    model = Glm5NextMTPModel.__new__(Glm5NextMTPModel)
    model.args = StartArgs()
    model.args.run_mode = "prefill"
    model.args.mtp_mode = "eagle_with_att"
    model.args.mtp_step = 1
    model.args.llm_kv_type = "fp8kv_dsa"
    model.tp_world_size_ = model.layers_num = 1
    model.prefill_graph = None
    model.layers_infer = [SimpleNamespace(is_linear_attention_layer=False, use_mhc=False)]
    model.trans_layers_weight = [object()]
    return model


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_mode", "decode"),
        ("run_mode", "normal"),
        ("mtp_mode", "eagle_no_att"),
        ("mtp_step", 3),
        ("llm_kv_type", "none"),
        ("enable_tpsp_mix_mode", True),
        ("enable_dp_prefill_balance", True),
        ("enable_prefill_decode_mixed", True),
        ("enable_prefill_microbatch_overlap", True),
    ],
)
def test_unsupported_execution_uses_full_forward(field, value):
    model = make_model()
    assert model._can_fill_prefill_kv_only()
    setattr(model.args, field, value)
    model.forward = Mock(return_value=object())
    model_input = SimpleNamespace(is_prefill=True)
    assert model.fill_prefill_kv(model_input) is model.forward.return_value
    model.forward.assert_called_once_with(model_input)


@pytest.mark.parametrize("field,value", [("tp_world_size_", 2), ("prefill_graph", object()), ("layers_num", 2)])
def test_unsupported_layout_uses_full_forward(field, value):
    model = make_model()
    setattr(model, field, value)
    assert not model._can_fill_prefill_kv_only()


@pytest.mark.parametrize("tokens", [0, 1, 8192])
@pytest.mark.parametrize("fail_write", [False, True])
def test_fill_pads_empty_ranks_and_releases_allocator(monkeypatch, tokens, fail_write):
    from lightllm.models.glm5_next_mtp import model as model_module

    model = make_model()
    model_input = SimpleNamespace(is_prefill=True, input_ids=torch.empty(tokens), to_cuda=Mock())
    infer = SimpleNamespace(
        input_ids=object(), init_some_extra_state=Mock(), init_att_state=Mock(), mtp_draft_input_hiddens=object()
    )
    model._create_padded_prefill_model_input = Mock(return_value=model_input)
    model._create_inferstate = Mock(return_value=infer)
    model.pre_infer = SimpleNamespace(context_forward=Mock(return_value=object()))
    model.pre_post_weight = object()
    writer = Mock(side_effect=RuntimeError("cache write failed") if fail_write else None)
    model.layers_infer[0].write_draft_prefill_cache = writer
    allocator = SimpleNamespace(cache_env_in=Mock(), cache_env_out=Mock())
    monkeypatch.setattr(model_module, "g_cache_manager", allocator)
    if fail_write:
        with pytest.raises(RuntimeError, match="cache write failed"):
            model.fill_prefill_kv(model_input)
    else:
        assert model.fill_prefill_kv(model_input) is None
    model._create_padded_prefill_model_input.assert_called_once_with(
        model_input=model_input, new_handle_token_num=max(1, tokens)
    )
    model.pre_infer.context_forward.assert_called_once_with(infer.input_ids, infer, model.pre_post_weight)
    writer.assert_called_once_with(model.pre_infer.context_forward.return_value, infer, model.trans_layers_weight[0])
    assert infer.mtp_draft_input_hiddens is None
    allocator.cache_env_in.assert_called_once()
    allocator.cache_env_out.assert_called_once()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "requests",
    [((3, 0, 1),), ((0, 0, 8192),), ((0, 131071, 129), (2, 1048573, 3)), ((1, 3, 1), (0, 5, 7))],
)
def test_cache_only_matches_full_prefill_cache_writes(monkeypatch, requests):
    from lightllm.models.glm5_next import indexer as indexer_module

    torch.manual_seed(53)
    layer_id, tokens = 45, sum(q for _, _, q in requests)
    layer = Glm5NextTransformerLayerInfer.__new__(Glm5NextTransformerLayerInfer)
    layer.layer_num_, layer.num_hidden_layers = layer_id, 1
    layer.use_mhc = layer.is_linear_attention_layer = False
    layer.tp_world_size_ = layer.tp_q_head_num_ = 1
    layer.embed_dim_ = layer.q_lora_rank = 64
    layer.kv_lora_rank = layer.qk_nope_head_dim = 512
    layer.eps_ = 1e-5
    layer.indexer = Glm5NextNsaInfer(
        layer_id, dict(index_topk=2048, index_n_heads=1, index_head_dim=128, rms_norm_eps=1e-5, index_kpool=4), 1
    )

    def projection(rows, cols):
        weight = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) * 0.1
        return SimpleNamespace(mm=Mock(side_effect=lambda x: x @ weight))

    def norm(input, eps, **_):
        return rmsnorm_forward(input, None, eps)

    weights = SimpleNamespace(
        att_norm_weight_=norm,
        qkv_a_proj_with_mqa_=projection(64, 576),
        q_a_layernorm_=SimpleNamespace(weight=torch.ones(64, device="cuda", dtype=torch.bfloat16)),
        kv_a_layernorm_=SimpleNamespace(weight=torch.ones(512, device="cuda", dtype=torch.bfloat16)),
        q_b_proj_=projection(64, 512),
        wk_proj_=projection(64, 128),
        k_norm_=norm,
        index_kpool_compress_gate=projection(64, 128),
        index_kpool_compress_ape=SimpleNamespace(weight=torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)),
    )
    resources = AttentionStateManager(request_capacity=4)
    stores = []
    for role in ("main", "draft0"):
        full = FP8E8M0PackedMLAState(FP8E8M0PackedMLAConfig(logical_layer_groups=((layer_id,),)))
        full.allocate_storage(2 * tokens + 2)
        full.storage.fill_(0x5A)
        tail = WindowCompressorState(
            WindowCompressorStateConfig((layer_id,), 4, 4, 128, "indexer", torch.bfloat16), 4, mtp_step=1
        )
        tail.buffer.normal_()
        resources.register_state(role, full)
        resources.register_state(role, tail)
        stores.extend((full.storage, tail.buffer))
    resources.finish_state_registration()
    initial = [s.clone() for s in stores]
    q_lengths = [q for _, _, q in requests]
    lengths = torch.cat([torch.arange(past + 1, past + q + 1, device="cuda") for _, past, q in requests]).int()
    infer = SimpleNamespace(
        is_prefill=True,
        is_draft_model=True,
        need_dp_prefill_balance=False,
        att_state_role="draft0",
        att_state_manager=resources,
        mem_index=2 * torch.arange(tokens, device="cuda", dtype=torch.int32) + 1,
        b_req_idx=torch.tensor([r for r, _, _ in requests], device="cuda", dtype=torch.int32),
        b_seq_len=torch.tensor([p + q for _, p, q in requests], device="cuda", dtype=torch.int32),
        b1_cu_q_seq_len=torch.tensor([0, *q_lengths], device="cuda", dtype=torch.int32).cumsum(0).int(),
        max_q_seq_len=max(q_lengths),
        max_kv_seq_len=2048,
        prefill_att_state=SimpleNamespace(q_kv_seq_len=lengths, q_req_idx=None),
        req_manager=SimpleNamespace(req_to_token_indexs=None),
    )
    # Top-k outputs are discarded here; run the normal projections and cache writers.
    monkeypatch.setattr(indexer_module, "expand_topk", Mock(return_value=(None, None)))
    hidden = torch.randn(tokens, 64, device="cuda", dtype=torch.bfloat16)
    _, kv = layer._get_qkv(layer._att_norm(hidden, infer, weights), infer, weights)
    layer._post_cache_kv(kv, infer, weights)
    layer.indexer._get_indices(
        **infer.get_topk_indices_params, infer_state=infer, att_state=infer.prefill_att_state, layer_weight=weights
    )
    expected = [s.clone() for s in stores]
    assert not torch.equal(initial[2], expected[2])
    for store, before in zip(stores, initial):
        store.copy_(before)
    weights.q_b_proj_.mm.reset_mock()
    layer.write_draft_prefill_cache(hidden, infer, weights)
    weights.q_b_proj_.mm.assert_not_called()
    for actual, reference in zip(stores, expected):
        assert torch.equal(actual, reference)
    for actual, before in zip(stores[:2], initial[:2]):
        assert torch.equal(actual, before)
    assert torch.equal(stores[2][:, ::2], initial[2][:, ::2])
    assert torch.equal(stores[3][:, 3], initial[3][:, 3])
