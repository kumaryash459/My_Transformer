from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """
    Configuration for the encoder-decoder Transformer.

    The default values are intentionally small enough to train
    on a 6 GB GPU while still representing the complete architecture.
    """

    # Vocabulary
    src_vocab_size: int = 10_000
    tgt_vocab_size: int = 10_000

    # Sequence
    max_seq_length: int = 128

    # Model dimensions
    d_model: int = 256
    num_heads: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4

    # Feed-forward network
    d_ff: int = 1024

    # Regularization
    dropout: float = 0.1

    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01

    # Hardware
    device: str = "cuda"

    def __post_init__(self):
        """Validate the configuration."""

        if self.d_model % self.num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads."
            )

        if self.dropout < 0.0 or self.dropout > 1.0:
            raise ValueError(
                "dropout must be between 0 and 1."
            )

        if self.d_model <= 0:
            raise ValueError(
                "d_model must be positive."
            )

        if self.num_heads <= 0:
            raise ValueError(
                "num_heads must be positive."
            )

        if self.num_encoder_layers <= 0:
            raise ValueError(
                "num_encoder_layers must be positive."
            )

        if self.num_decoder_layers <= 0:
            raise ValueError(
                "num_decoder_layers must be positive."
            )