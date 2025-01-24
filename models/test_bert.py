import torch
import unittest
from bert import BERTBasedModel, BertModelConfig

class TestEncoderBehavior(unittest.TestCase):
    def setUp(self):
        # Create a dummy config and model
        self.config = BertModelConfig(
            name="bert",
            seq_input_dim=128,
            go_input_dim=64,
            hidden_dim=256,
            num_encoder_layers=2,
            num_decoder_layers=2,
            num_attention_heads=4,
            intermediate_size=512,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            hidden_act="gelu",
            layer_norm_eps=1e-12,
            decoder=False,
        )
        self.model = BERTBasedModel(self.config)
        self.model.eval()  # Ensure the model is in evaluation mode

    def test_positional_dependence_without_mask(self):
        # Create dummy inputs
        self.model.config.decoder = True
        batch_size = 2
        seq_len = 10
        go_len = 5
        seq_emb = torch.randn(batch_size, seq_len, self.config.seq_input_dim)
        go_emb = torch.randn(batch_size, go_len, self.config.go_input_dim)
        attention_mask_seq_emb = torch.ones(batch_size, seq_len)

        # Get full sequence output
        full_output = self.model(go_emb, seq_emb, None, attention_mask_seq_emb)

        # Test positional dependence for random shuffling of GO embeddings
        shuffle = torch.randperm(go_len)
        go_emb_shuffled = go_emb[:, shuffle, :]
        shuffled_output = self.model(go_emb_shuffled, seq_emb, None, attention_mask_seq_emb)
        # Invert the shuffle by using argsort to get inverse permutation
        inverse_shuffle = torch.argsort(shuffle)
        shuffled_output = shuffled_output[:, inverse_shuffle, :]

        self.assertTrue(
            torch.allclose(
                go_emb_shuffled[:, inverse_shuffle, :], go_emb[:, :, :], atol=1e-5
            ),
            f"GO embeddings are not the same after shuffling.",
        )

        self.assertFalse(
            torch.allclose(
                full_output, shuffled_output, atol=1e-5
            ),
            f"Output is the same when shuffling GO embeddings.",
        )


    def test_positional_dependence_with_mask(self):
        # Create dummy inputs
        self.model.config.decoder = True
        batch_size = 2
        seq_len = 10
        go_len = 5
        seq_emb = torch.randn(batch_size, seq_len, self.config.seq_input_dim)
        go_emb = torch.randn(batch_size, go_len, self.config.go_input_dim)
        attention_mask_seq_emb = torch.ones(batch_size, seq_len)
        attention_mask_go_emb = torch.ones(batch_size, go_len)

        # Get full sequence output
        full_output = self.model(go_emb, seq_emb, attention_mask_go_emb, attention_mask_seq_emb)

        # Test positional dependence for random shuffling of GO embeddings
        shuffle = torch.randperm(go_len)
        go_emb_shuffled = go_emb[:, shuffle, :]
        shuffled_output = self.model(go_emb_shuffled, seq_emb, attention_mask_go_emb, attention_mask_seq_emb)
        # Invert the shuffle by using argsort to get inverse permutation
        inverse_shuffle = torch.argsort(shuffle)
        shuffled_output = shuffled_output[:, inverse_shuffle, :]

        self.assertTrue(
            torch.allclose(
                go_emb_shuffled[:, inverse_shuffle, :], go_emb[:, :, :], atol=1e-5
            ),
            f"GO embeddings are not the same after shuffling.",
        )

        self.assertFalse(
            torch.allclose(
                full_output, shuffled_output, atol=1e-5
            ),
            f"Output is the same when shuffling GO embeddings.",
        )

    def test_positional_independence(self):
        # Create dummy inputs
        self.model.config.decoder = False
        batch_size = 2
        seq_len = 10
        go_len = 5
        seq_emb = torch.randn(batch_size, seq_len, self.config.seq_input_dim)
        go_emb = torch.randn(batch_size, go_len, self.config.go_input_dim)
        attention_mask_seq_emb = torch.ones(batch_size, seq_len)
        attention_mask_go_emb = torch.ones(batch_size, go_len)

        # Get full sequence output
        full_output = self.model(go_emb, seq_emb, attention_mask_go_emb, attention_mask_seq_emb)

        # Test positional independence for random shuffling of GO embeddings

        shuffle = torch.randperm(go_len)
        go_emb_shuffled = go_emb[:, shuffle, :]
        shuffled_output = self.model(go_emb_shuffled, seq_emb, attention_mask_go_emb, attention_mask_seq_emb)
        # Invert the shuffle by using argsort to get inverse permutation
        inverse_shuffle = torch.argsort(shuffle)
        shuffled_output = shuffled_output[:, inverse_shuffle, :]

        self.assertTrue(
            torch.allclose(
                go_emb_shuffled[:, inverse_shuffle, :], go_emb[:, :, :], atol=1e-5
            ),
            f"GO embeddings are not the same after shuffling.",
        )
        # Compare outputs
        self.assertTrue(
            torch.allclose(
                full_output, shuffled_output, atol=1e-5
            ),
            f"Output changes when shuffling GO embeddings.",
        )

    def test_positional_independence_without_mask(self):
        # Create dummy inputs
        self.model.config.decoder = False
        batch_size = 2
        seq_len = 10
        go_len = 5
        seq_emb = torch.randn(batch_size, seq_len, self.config.seq_input_dim)
        go_emb = torch.randn(batch_size, go_len, self.config.go_input_dim)
        attention_mask_seq_emb = torch.ones(batch_size, seq_len)

        # Get full sequence output
        full_output = self.model(go_emb, seq_emb, None, attention_mask_seq_emb)

        # Test positional independence for random shuffling of GO embeddings

        shuffle = torch.randperm(go_len)
        go_emb_shuffled = go_emb[:, shuffle, :]
        shuffled_output = self.model(go_emb_shuffled, seq_emb, None, attention_mask_seq_emb)
        # Invert the shuffle by using argsort to get inverse permutation
        inverse_shuffle = torch.argsort(shuffle)
        shuffled_output = shuffled_output[:, inverse_shuffle, :]

        self.assertTrue(
            torch.allclose(
                go_emb_shuffled[:, inverse_shuffle, :], go_emb[:, :, :], atol=1e-5
            ),
            f"GO embeddings are not the same after shuffling.",
        )
        # Compare outputs
        self.assertTrue(
            torch.allclose(
                full_output, shuffled_output, atol=1e-5
            ),
            f"Output changes when shuffling GO embeddings.",
        )

    def test_no_decoder_attributes(self):
        # Check that the model does not have decoder-specific attributes
        self.assertFalse(hasattr(self.model.decoder, "past_key_values"), "Model decoder has 'past_key_values' attribute, indicating decoder behavior.")
