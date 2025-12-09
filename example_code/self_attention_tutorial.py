import torch.nn as nn
import torch

# ====================================================================
# CORE ATTENTION FUNCTIONS (Used by both Single-Head and Multi-Head)
# ====================================================================
def _calculate_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # TODO: Implement Q * K^T
    pass

def _scale_and_softmax(scores: torch.Tensor, d_k: int) -> torch.Tensor:
    # TODO: Implement scaling and Softmax
    return nn.functional.softmax(scores / d_k**0.5, dim=-1)


def _weighted_sum(attn_weights: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Calculates the final attention output: Output = Weights * V.
    
    Input shapes: Weights (..., S, S), V (..., S, d_v or d_k)
    Output shape: (..., S, d_v or d_k)
    """
    # Batch matrix multiplication: (..., S, S) @ (..., S, d_v) -> (..., S, d_v)
    # TODO: Implement the batch matrix multiplication
    return attn_weights.matmul(v)


# ====================================================================
# SINGLE-HEAD SELF-ATTENTION MODULE
# ====================================================================

class SelfAttention(nn.Module):
    """
    Single-Head Self-Attention Mechanism implemented from scratch.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Linear projections for Q, K, V (all map embed_dim to embed_dim)
        # TODO: Define the linear layers W_q, W_k, W_v
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # d_k is the dimension of the Key vectors
        self.d_k = embed_dim 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Linear Projections: (B, S, D_embed)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        
        # 2. Calculate Attention Scores: (B, S, S)
        scores = _calculate_scores(q, k)
        
        # 3. Scaling and Softmax: (B, S, S)
        attn_weights = _scale_and_softmax(scores, self.d_k)
        
        # 4. Weighted Sum: (B, S, D_embed)
        output = _weighted_sum(attn_weights, v)
        
        return output   


# ====================================================================
# MULTI-HEAD UTILITY FUNCTIONS
# ====================================================================

def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Reshapes the tensor to separate the heads for parallel computation.
    (B, S, D_model) -> (B, H, S, d_k)
    """
    B, S, D_model = x.shape
    d_k = D_model // num_heads
    
    # 1. Reshape: (B, S, D_model) -> (B, S, H, d_k)
    x_reshaped = x.view(B, S, num_heads,d_k)
    
    # 2. Transpose: (B, S, H, d_k) -> (B, H, S, d_k)
    x_split = x_reshaped.transpose(1, 2)
    
    return x_split

def _combine_heads(x_split: torch.Tensor) -> torch.Tensor:
    """
    Concatenates the head outputs and flattens them back into D_model.
    (B, H, S, d_k) -> (B, S, D_model)
    """
    B, H, S, d_k = x_split.shape
    D_model = H * d_k
    
    # 1. Transpose back: (B, H, S, d_k) -> (B, S, H, d_k)
    x_transposed = x_split.transpose(1, 2)
    
    # 2. Reshape/Combine: (B, S, H, d_k) -> (B, S, D_model)
    x_combined = x_transposed.contiguous().view(B, S, D_model)
    
    return x_combined

# ====================================================================
# MULTI-HEAD SELF-ATTENTION MODULE
# ====================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention Mechanism implemented from scratch.
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.D_model = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads # Dimension of each head

        # Initial Projections (maps D_model to D_model)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Final Projection (maps combined D_model back to D_model)
        self.W_out = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D_model = x.shape
        H = self.num_heads

        # 1. Initial Linear Projections: (B, S, D_model)
        q_proj = self.W_q(x)
        k_proj = self.W_k(x)
        v_proj = self.W_v(x)
        
        # 2. Splitting Heads: (B, S, D_model) -> (B, H, S, d_k)
        q = _split_heads(q_proj, H)
        k = _split_heads(k_proj, H)
        v = _split_heads(v_proj, H)
        
        # 3. Parallel Attention Calculation
        
        # Scores: (B, H, S, S)
        scores = _calculate_scores(q, k)
        
        # Attention Weights: (B, H, S, S)
        attn_weights = _scale_and_softmax(scores, self.d_k)
        
        # Weighted Sum (Output per head): (B, H, S, d_k)
        attn_output_split = _weighted_sum(attn_weights, v)
        
        # 4. Combining Heads: (B, H, S, d_k) -> (B, S, D_model)
        attn_output_combined = _combine_heads(attn_output_split)
        
        # 5. Final Linear Projection: (B, S, D_model)
        output = self.W_out(attn_output_combined)
        
        return output
    
# ====================================================================
# TEST SUITE
# ====================================================================

def run_tests():
    print("--- Running Self-Attention Test Suite ---")
    
    # Setup common test parameters
    B, S, D = 2, 5, 4  # Batch, Sequence, Embedding Dim
    H = 2              # Heads for MHSA
    D_MHSA = 8         # Embedding Dim for MHSA (D_MHSA % H == 0)
    
    # ---------------------------------------------
    # 1. Single-Head Component Tests
    # ---------------------------------------------
    
    # Test 1.1: Scores Calculation
    q_single = torch.randn(B, S, D)
    k_single = torch.randn(B, S, D)
    scores_single = _calculate_scores(q_single, k_single)
    assert scores_single.shape == (B, S, S)
    print(f"✅ Test 1.1 (Scores Single): Shape {scores_single.shape}")

    # Test 1.2: Scaling and Softmax
    d_k_single = D
    attn_weights_single = _scale_and_softmax(scores_single, d_k_single)
    # Check if the sum along the last dimension is close to 1
    assert torch.allclose(attn_weights_single.sum(dim=-1), torch.ones(B, S))
    print(f"✅ Test 1.2 (Softmax Single): Sums to 1.0")

    # Test 1.3: Weighted Sum
    v_single = torch.randn(B, S, D)
    output_single_sum = _weighted_sum(attn_weights_single, v_single)
    assert output_single_sum.shape == (B, S, D)
    print(f"✅ Test 1.3 (Weighted Sum Single): Shape {output_single_sum.shape}")
    
    # Test 1.4: End-to-End Single-Head
    x_single = torch.randn(B, S, D)
    attn_scratch = SelfAttention(D)
    output_scratch = attn_scratch(x_single)
    assert output_scratch.shape == (B, S, D)
    print(f"✅ Test 1.4 (End-to-End Single): Shape {output_scratch.shape}")

    # ---------------------------------------------
    # 2. Multi-Head Component Tests
    # ---------------------------------------------
    
    # Test 2.1: Head Splitting
    x_mhsa = torch.randn(B, S, D_MHSA)
    x_split = _split_heads(x_mhsa, H)
    d_k_mhsa = D_MHSA // H
    assert x_split.shape == (B, H, S, d_k_mhsa)
    print(f"\n✅ Test 2.1 (Split Heads): Shape {x_split.shape}")

    # Test 2.2: Scores Calculation (Multi-Head Input)
    scores_mhsa = _calculate_scores(x_split, x_split)
    assert scores_mhsa.shape == (B, H, S, S)
    print(f"✅ Test 2.2 (Scores Multi): Shape {scores_mhsa.shape}")

    # Test 2.3: Head Combining
    x_combined = _combine_heads(x_split)
    assert x_combined.shape == (B, S, D_MHSA)
    print(f"✅ Test 2.3 (Combine Heads): Shape {x_combined.shape}")
    
    # Test 2.4: End-to-End Multi-Head
    attn_mhsa_scratch = MultiHeadSelfAttention(D_MHSA, H)
    output_mhsa_scratch = attn_mhsa_scratch(x_mhsa)
    assert output_mhsa_scratch.shape == (B, S, D_MHSA)
    
    # Compare shape against PyTorch built-in (for validation)
    attn_builtin = torch.nn.MultiheadAttention(
        embed_dim=D_MHSA, num_heads=H, batch_first=True
    )
    output_builtin, _ = attn_builtin(x_mhsa, x_mhsa, x_mhsa)
    assert output_mhsa_scratch.shape == output_builtin.shape
    print(f"✅ Test 2.4 (End-to-End Multi): Shape {output_mhsa_scratch.shape}")
    
    print("\n--- All Self-Attention Tests Passed Successfully! ---")


if __name__ == '__main__':
    run_tests()