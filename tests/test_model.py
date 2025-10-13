import pytest
import torch
from torch import nn

from gpoetry.model import MLP, GPTConfig, Head, MHSelfAttention, Transformer, GPTModel


class TestHead:
    """Tests for the Head (single attention head) module."""

    @pytest.fixture
    def head(self):
        """Create a Head instance for testing."""
        return Head(emb_dim=64, head_size=16, block_size=128, p=0.1)

    def test_initialization(self, head):
        """Test that Head initializes correctly."""
        assert isinstance(head.querie, nn.Linear)
        assert isinstance(head.key, nn.Linear)
        assert isinstance(head.value, nn.Linear)
        assert isinstance(head.dropout, nn.Dropout)
        assert head.tril.shape == (128, 128)
        assert torch.all(head.tril == torch.tril(torch.ones(128, 128)))

    def test_forward_shape(self, head):
        """Test that forward pass produces correct output shape."""
        batch_size, seq_len, emb_dim = 2, 10, 64
        x = torch.randn(batch_size, seq_len, emb_dim)
        out = head(x)
        assert out.shape == (batch_size, seq_len, 16)

    def test_forward_different_seq_lengths(self, head):
        """Test forward pass with various sequence lengths."""
        batch_size, emb_dim = 4, 64
        for seq_len in [1, 5, 32, 128]:
            x = torch.randn(batch_size, seq_len, emb_dim)
            out = head(x)
            assert out.shape == (batch_size, seq_len, 16)


class TestMHSelfAttention:
    """Tests for the Multi-Head Self-Attention module."""

    @pytest.fixture
    def mhsa(self):
        """Create a MHSelfAttention instance for testing."""
        return MHSelfAttention(
            emb_dim=64, num_heads=4, head_size=16, block_size=128, p=0.1
        )

    def test_initialization(self, mhsa):
        """Test that MHSelfAttention initializes correctly."""
        assert isinstance(mhsa.heads, nn.ModuleList)
        assert len(mhsa.heads) == 4
        assert all(isinstance(h, Head) for h in mhsa.heads)
        assert isinstance(mhsa.dropout, nn.Dropout)

    def test_forward_shape(self, mhsa):
        """Test that forward pass produces correct output shape."""
        batch_size, seq_len, emb_dim = 2, 10, 64
        x = torch.randn(batch_size, seq_len, emb_dim)
        out = mhsa(x)
        # Output should concatenate all heads: 4 heads * 16 head_size = 64
        assert out.shape == (batch_size, seq_len, 64)

    def test_gradient_flow(self, mhsa):
        """Test that gradients flow through the module."""
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = mhsa(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestMLP:
    """Tests for the MLP module."""

    @pytest.fixture
    def ffn(self):
        """Create a MLP instance for testing."""
        return MLP(emb_dim=64, p=0.1)

    def test_initialization(self, ffn):
        """Test that MLP initializes correctly."""
        assert isinstance(ffn.mlp, nn.Sequential)
        assert len(ffn.mlp) == 4
        assert isinstance(ffn.mlp[0], nn.Linear)
        assert isinstance(ffn.mlp[1], nn.GELU)
        assert isinstance(ffn.mlp[2], nn.Linear)
        assert isinstance(ffn.mlp[3], nn.Dropout)

    def test_expansion_ratio(self, ffn):
        """Test that the hidden dimension expands by 4x."""
        # First linear layer should expand from emb_dim to 4*emb_dim
        assert ffn.mlp[0].out_features == 256  # 64 * 4
        # Second linear layer should project back to emb_dim
        assert ffn.mlp[2].out_features == 64

    def test_forward_shape(self, ffn):
        """Test that forward pass maintains input shape."""
        batch_size, seq_len, emb_dim = 2, 10, 64
        x = torch.randn(batch_size, seq_len, emb_dim)
        out = ffn(x)
        assert out.shape == x.shape

    def test_gradient_flow(self, ffn):
        """Test that gradients flow through the module."""
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = ffn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestTransformer:
    """Tests for the Transformer module."""

    @pytest.fixture
    def layer(self):
        """Create a Transformer instance for testing."""
        return Transformer(emb_dim=64, num_heads=4, block_size=128, p=0.1)

    def test_initialization(self, layer):
        """Test that Transformer initializes correctly."""
        assert isinstance(layer.mhsa, MHSelfAttention)
        assert isinstance(layer.ln1, nn.LayerNorm)
        assert isinstance(layer.ffn, MLP)
        assert isinstance(layer.ln2, nn.LayerNorm)

    def test_forward_shape(self, layer):
        """Test that forward pass maintains input shape."""
        batch_size, seq_len, emb_dim = 2, 10, 64
        x = torch.randn(batch_size, seq_len, emb_dim)
        out = layer(x)
        assert out.shape == x.shape

    def test_residual_connections(self, layer):
        """Test that residual connections are working."""
        x = torch.randn(2, 10, 64)
        out = layer(x)
        # Output should be different from input but related through residuals
        assert not torch.allclose(out, x)
        # The norm of output shouldn't be drastically different
        assert out.std() / x.std() < 10

    def test_gradient_flow(self, layer):
        """Test that gradients flow through the entire layer."""
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestGPTModel:
    """Tests for the main GPTModel model."""

    @pytest.fixture
    def model(self):
        """Create a GPTModel model instance for testing."""
        config = GPTConfig(
            vocab_size=1000,
            num_heads=4,
            num_layers=2,
            block_size=128,
            emb_dim=64,
            p=0.1,
        )
        return GPTModel(config=config)

    def test_initialization(self, model):
        """Test that GPTModel initializes correctly."""
        assert isinstance(model.token_emb, nn.Embedding)
        assert isinstance(model.pos_emb, nn.Embedding)
        assert isinstance(model.blocks, nn.Sequential)
        assert isinstance(model.ln, nn.LayerNorm)
        assert isinstance(model.lm_head, nn.Linear)
        assert len(model.blocks) == 2

    def test_embedding_dimensions(self, model):
        """Test that embeddings have correct dimensions."""
        assert model.token_emb.num_embeddings == 1000
        assert model.token_emb.embedding_dim == 64
        assert model.pos_emb.num_embeddings == 128
        assert model.pos_emb.embedding_dim == 64

    def test_forward_shape(self, model):
        """Test that forward pass produces correct output shape."""
        batch_size, seq_len = 4, 20
        x = torch.randint(0, 1000, (batch_size, seq_len))
        logits = model(x)
        assert logits.shape == (batch_size, seq_len, 1000)

    def test_forward_different_batch_sizes(self, model):
        """Test forward pass with various batch sizes."""
        seq_len = 15
        for batch_size in [1, 2, 8, 16]:
            x = torch.randint(0, 1000, (batch_size, seq_len))
            logits = model(x)
            assert logits.shape == (batch_size, seq_len, 1000)

    def test_forward_different_seq_lengths(self, model):
        """Test forward pass with various sequence lengths."""
        batch_size = 4
        for seq_len in [1, 10, 50, 128]:
            x = torch.randint(0, 1000, (batch_size, seq_len))
            logits = model(x)
            assert logits.shape == (batch_size, seq_len, 1000)

    def test_gradient_flow(self, model):
        """Test that gradients flow through the entire model."""
        x = torch.randint(0, 1000, (2, 10))
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        # Check that embeddings received gradients
        assert model.token_emb.weight.grad is not None
        assert model.pos_emb.weight.grad is not None

    def test_output_distribution(self, model):
        """Test that output logits have reasonable distribution."""
        x = torch.randint(0, 1000, (4, 20))
        logits = model(x)

        # Logits should not be all the same
        assert logits.std() > 0.1

        # Logits should not have extreme values
        assert logits.abs().max() < 100

    def test_positional_embedding_bounds(self, model):
        """Test that positional embeddings don't exceed block_size."""
        # This should work fine
        x = torch.randint(0, 1000, (2, 128))
        logits = model(x)
        assert logits.shape == (2, 128, 1000)

    def test_deterministic_output(self, model):
        """Test that model produces deterministic output in eval mode."""
        model.eval()
        x = torch.randint(0, 1000, (2, 10))

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert torch.allclose(out1, out2)

    def test_training_mode_randomness(self, model):
        """Test that dropout introduces randomness in training mode."""
        model.train()
        x = torch.randint(0, 1000, (2, 10))

        out1 = model(x)
        out2 = model(x)

        # Outputs should be different due to dropout
        assert not torch.allclose(out1, out2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
