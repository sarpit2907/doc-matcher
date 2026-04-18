"""
GNN Graph Transformer — Smoke Test & Verification Script
=========================================================

Verifies that the GraphTransformerBlock and its integration into
LineLightGlue work correctly. Run this before full inference to
catch any issues early.

Usage:
    python verify_gnn.py
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, "src")

def test_graph_builder():
    """Test LineGraphBuilder produces correct shapes."""
    from line_matching.line_lightglue.graph_transformer import LineGraphBuilder

    builder = LineGraphBuilder(k_neighbors=5, self_loops=True)

    # Simulate batch of line descriptors
    B, N, D = 2, 20, 256
    descriptors = torch.randn(B, N, D)

    adj_matrix, edge_features = builder.build_graph(descriptors)

    assert adj_matrix.shape == (B, N, N), f"adj_matrix shape: {adj_matrix.shape}"
    assert adj_matrix.dtype == torch.bool, f"adj_matrix dtype: {adj_matrix.dtype}"
    assert edge_features.shape == (B, N, N, 4), f"edge_features shape: {edge_features.shape}"

    # Check symmetry
    assert (adj_matrix == adj_matrix.transpose(1, 2)).all(), "adj_matrix not symmetric"

    # Check self-loops
    for b in range(B):
        assert adj_matrix[b].diagonal().all(), "Missing self-loops"

    # Check edge feature ranges
    sim = edge_features[..., 0]  # cosine similarity should be [-1, 1]
    assert sim.min() >= -1.01 and sim.max() <= 1.01, f"Cosine sim out of range: [{sim.min()}, {sim.max()}]"

    print("  ✓ LineGraphBuilder: shapes, symmetry, self-loops, feature ranges OK")


def test_graph_transformer_block():
    """Test GraphTransformerBlock produces correct output shapes."""
    from line_matching.line_lightglue.graph_transformer import GraphTransformerBlock

    block = GraphTransformerBlock(
        embed_dim=256,
        num_heads=4,
        edge_dim=4,
        k_neighbors=5,
        use_sparse_attention=False,
    )

    B, N, D = 2, 20, 256
    x = torch.randn(B, N, D)
    out = block(x)

    assert out.shape == (B, N, D), f"Output shape: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output!"

    # Test with mask
    mask = torch.ones(B, N, N, dtype=torch.bool)
    out_masked = block(x, mask=mask)
    assert out_masked.shape == (B, N, D), f"Masked output shape: {out_masked.shape}"

    # Test graph info caching
    info = block.get_graph_info()
    assert info is not None, "get_graph_info() returned None after forward"
    assert 'adj_matrix' in info, "Missing adj_matrix"
    assert 'edge_features' in info, "Missing edge_features"
    assert 'attn_weights' in info, "Missing attn_weights"
    assert 'graph_gate' in info, "Missing graph_gate"

    print("  ✓ GraphTransformerBlock: output shape, NaN check, masking, graph info OK")


def test_sparse_attention():
    """Test GraphTransformerBlock with sparse attention masking."""
    from line_matching.line_lightglue.graph_transformer import GraphTransformerBlock

    block = GraphTransformerBlock(
        embed_dim=256,
        num_heads=4,
        edge_dim=4,
        k_neighbors=3,
        use_sparse_attention=True,  # Enable sparse masking
    )

    B, N, D = 1, 15, 256
    x = torch.randn(B, N, D)
    out = block(x)

    assert out.shape == (B, N, D), f"Sparse output shape: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in sparse output!"

    # Check that attention weights are sparse (many zeros)
    info = block.get_graph_info()
    attn = info['attn_weights']
    sparsity = (attn < 1e-6).float().mean().item()
    assert sparsity > 0.3, f"Expected sparse attention, got sparsity={sparsity:.2f}"

    print(f"  ✓ Sparse attention: sparsity={sparsity:.2f} (expected >0.3)")


def test_weight_compatibility():
    """Test that pretrained LineSelfBlock weights load into GraphTransformerBlock."""
    from line_matching.line_lightglue.line_lightglue_model import LineSelfBlock
    from line_matching.line_lightglue.graph_transformer import GraphTransformerBlock

    # Create original block and get its state dict
    original = LineSelfBlock(embed_dim=256, num_heads=4)
    original_state = original.state_dict()

    # Create graph transformer block
    graph = GraphTransformerBlock(embed_dim=256, num_heads=4)

    # Load original weights with strict=False
    missing, unexpected = graph.load_state_dict(original_state, strict=False)

    # Check that core weights loaded
    loaded_keys = set(original_state.keys())
    graph_keys = set(graph.state_dict().keys())
    core_keys = loaded_keys & graph_keys

    assert len(core_keys) == len(original_state), \
        f"Not all original keys matched: {loaded_keys - graph_keys}"

    # Check that new params are in the missing list
    new_params = graph_keys - loaded_keys
    expected_new = {'edge_encoder.0.weight', 'edge_encoder.0.bias',
                    'edge_encoder.2.weight', 'edge_encoder.2.bias',
                    'graph_gate'}
    assert new_params == expected_new, f"Unexpected new params: {new_params - expected_new}"

    print(f"  ✓ Weight compatibility: {len(core_keys)} original params loaded, "
          f"{len(new_params)} new params (expected)")


def test_config_passthrough():
    """Test that config parameters are properly passed to GraphTransformerBlock."""
    from line_matching.line_lightglue.line_lightglue_model import LineLightGlue

    # Test with custom config
    conf = {
        "use_graph_transformer": True,
        "graph_k_neighbors": 8,
        "graph_edge_dim": 4,
        "graph_sparse_attention": True,
    }
    model = LineLightGlue(conf)

    # Check that the first transformer layer has the right config
    layer = model.transformers[0]
    graph_block = layer.self_attn

    assert hasattr(graph_block, 'graph_builder'), "Missing graph_builder"
    assert graph_block.graph_builder.k_neighbors == 8, \
        f"k_neighbors={graph_block.graph_builder.k_neighbors}, expected 8"
    assert graph_block.use_sparse_attention == True, \
        f"use_sparse_attention={graph_block.use_sparse_attention}, expected True"

    # Test with GNN disabled
    conf_off = {"use_graph_transformer": False}
    model_off = LineLightGlue(conf_off)
    layer_off = model_off.transformers[0]
    assert not hasattr(layer_off.self_attn, 'graph_builder'), \
        "graph_builder should not exist when GNN is disabled"

    print("  ✓ Config passthrough: k_neighbors, sparse_attention, GNN toggle OK")


def test_gradient_flow():
    """Test that gradients flow through the graph bias path."""
    from line_matching.line_lightglue.graph_transformer import GraphTransformerBlock

    block = GraphTransformerBlock(embed_dim=256, num_heads=4)

    B, N, D = 1, 10, 256
    x = torch.randn(B, N, D, requires_grad=True)
    out = block(x)
    loss = out.sum()
    loss.backward()

    # Check gradient on input
    assert x.grad is not None, "No gradient on input"
    assert not torch.isnan(x.grad).any(), "NaN in input gradient"

    # Check gradient on graph gate
    assert block.graph_gate.grad is not None, "No gradient on graph_gate"
    assert not torch.isnan(block.graph_gate.grad).any(), "NaN in graph_gate gradient"

    # Check gradient on edge encoder
    for name, param in block.edge_encoder.named_parameters():
        assert param.grad is not None, f"No gradient on edge_encoder.{name}"
        assert not torch.isnan(param.grad).any(), f"NaN in edge_encoder.{name} gradient"

    print("  ✓ Gradient flow: input, graph_gate, edge_encoder all receive gradients")


def test_parameter_count():
    """Count new parameters introduced by GNN."""
    from line_matching.line_lightglue.line_lightglue_model import LineSelfBlock
    from line_matching.line_lightglue.graph_transformer import GraphTransformerBlock

    original = LineSelfBlock(embed_dim=256, num_heads=4)
    graph = GraphTransformerBlock(embed_dim=256, num_heads=4)

    orig_params = sum(p.numel() for p in original.parameters())
    graph_params = sum(p.numel() for p in graph.parameters())
    new_params = graph_params - orig_params

    # Per layer: edge_encoder = Linear(4,8) + Linear(8,4) + gate = 4*8+8 + 8*4+4 + 1 = 77
    # Actually: Linear(4, 2*4=8): 4*8+8=40, Linear(8, 4): 8*4+4=36, gate=1 → total=77
    # Wait, num_heads=4, so Linear(4, 8) → Linear(8, 4) → 40+36+1=77... let me check
    # edge_encoder: Linear(4, 2*4=8): w=32, b=8 → 40; Linear(8, 4): w=32, b=4 → 36; gate=1
    # Total per layer = 77

    print(f"  ✓ Parameter count: original={orig_params:,}, graph={graph_params:,}, "
          f"new per layer={new_params:,}, "
          f"total new (9 layers)={new_params*9:,}")


def test_gpu_if_available():
    """Test on GPU if available."""
    if not torch.cuda.is_available():
        print("  ⚠ GPU not available, skipping GPU test")
        return

    from line_matching.line_lightglue.graph_transformer import GraphTransformerBlock

    device = torch.device("cuda:0")
    block = GraphTransformerBlock(embed_dim=256, num_heads=4).to(device)

    B, N, D = 2, 25, 256
    x = torch.randn(B, N, D, device=device)
    out = block(x)

    assert out.device.type == "cuda", f"Output on wrong device: {out.device}"
    assert out.shape == (B, N, D)
    assert not torch.isnan(out).any()

    print(f"  ✓ GPU test: forward pass on {torch.cuda.get_device_name(0)} OK")


if __name__ == "__main__":
    print("=" * 60)
    print("GNN Graph Transformer — Verification Tests")
    print("=" * 60)

    tests = [
        ("1. Graph Builder", test_graph_builder),
        ("2. Graph Transformer Block", test_graph_transformer_block),
        ("3. Sparse Attention", test_sparse_attention),
        ("4. Weight Compatibility", test_weight_compatibility),
        ("5. Config Passthrough", test_config_passthrough),
        ("6. Gradient Flow", test_gradient_flow),
        ("7. Parameter Count", test_parameter_count),
        ("8. GPU (if available)", test_gpu_if_available),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n{name}:")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    sys.exit(1 if failed > 0 else 0)
