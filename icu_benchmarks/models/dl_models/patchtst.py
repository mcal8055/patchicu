import gin
import torch
from torch import nn

from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.wrappers import DLPredictionWrapper
from transformers import PatchTSTModel, PatchTSTConfig


@gin.configurable
class PatchTST(DLPredictionWrapper):
    """HuggingFace PatchTST adapted for YAIB per-timestep classification.

    Uses the base PatchTSTModel to get patch-level hidden states, aggregates
    across channels (channel-independence), maps patches back to timestep
    resolution via repeat_interleave, then classifies per-timestep.

    Data flow:
        Input:  [batch, time, features]
        -> PatchTSTModel: [batch, features, num_patches, d_model]
        -> mean over features: [batch, num_patches, d_model]
        -> repeat_interleave(patch_stride): [batch, ~time, d_model]
        -> self.logit: [batch, time, num_classes]
    """

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        input_size,
        num_classes,
        patch_length=12,
        patch_stride=12,
        d_model=128,
        num_attention_heads=4,
        num_hidden_layers=3,
        ffn_dim=256,
        dropout=0.1,
        head_dropout=0.0,
        *args,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            num_classes=num_classes,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=d_model,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            head_dropout=head_dropout,
            *args,
            **kwargs,
        )

        # d_model must be BOTH even (for sin/cos positional encoding) AND
        # divisible by num_attention_heads. The two constraints aren't
        # independent when num_attention_heads is odd (e.g. 3), so bump up
        # until both hold.
        while d_model % 2 != 0 or d_model % num_attention_heads != 0:
            d_model += 1

        # Use full padded sequence length as context
        seq_length = input_size[1]

        # Validate patch geometry
        if patch_length > seq_length:
            raise ValueError(f"patch_length ({patch_length}) must be <= sequence length ({seq_length})")
        num_patches = (seq_length - patch_length) // patch_stride + 1
        if num_patches < 1:
            raise ValueError(f"patch_length={patch_length}, patch_stride={patch_stride}, seq_length={seq_length} produces 0 patches")

        # Store for use in forward()
        self.patch_stride = patch_stride

        # Build HuggingFace PatchTST config.
        #
        # Critical: HF PatchTSTConfig has a generic ``dropout`` field that is
        # stored but never read by any model component, plus six specific
        # fields (``attention_dropout``, ``positional_dropout``, ``path_dropout``,
        # ``ff_dropout``, ``head_dropout``) that actually drive Dropout module
        # creation. Earlier versions of this wrapper only set the inert generic
        # ``dropout`` and ``head_dropout``, leaving all four specific fields at
        # their 0.0 default — so encoder/attention/FFN dropout silently never
        # ran during training, MC dropout was deterministic, and Optuna's
        # ``dropout`` sweep was inert. Wire the gin ``dropout`` parameter into
        # all four specific fields so the encoder actually regularizes.
        config = PatchTSTConfig(
            num_input_channels=input_size[2],   # number of clinical features
            context_length=seq_length,           # full padded sequence length
            patch_length=patch_length,           # timesteps per patch
            patch_stride=patch_stride,           # stride between patches
            d_model=d_model,                     # transformer hidden dim
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,                     # kept for completeness; HF ignores it
            head_dropout=head_dropout,           # used by HF prediction heads (we bypass)
            attention_dropout=dropout,
            positional_dropout=dropout,
            path_dropout=dropout,
            ff_dropout=dropout,
            attn_implementation=__import__("os").environ.get("PATCHTST_ATTN", "eager"),
        )

        # The HF base model: patches -> transformer -> patch-level hidden states
        self.model = PatchTSTModel(config)

        # YAIB uses its own classification head (``self.logit`` below) rather
        # than HF's PatchTSTPredictionHead, which is the only place ``head_dropout``
        # is consumed by the HF model. Apply head_dropout here ourselves so the
        # gin parameter has somewhere to act.
        self.head_dropout = (
            nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        )

        # Per-timestep classification head
        # Also satisfies YAIB requirement: set_metrics() reads self.logit.out_features (wrappers.py:282)
        self.logit = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: [batch, time, features]
        batch_size, seq_len, n_features = x.shape

        # Single forward pass through PatchTST base model
        # Input: (batch, seq_length, num_input_channels) — matches YAIB format
        outputs = self.model(past_values=x)

        # last_hidden_state: [batch, num_input_channels, num_patches, d_model]
        hidden = outputs.last_hidden_state

        # Aggregate across channels (features) — channel independence means each
        # feature was processed independently, now we combine them
        hidden = hidden.mean(dim=1)  # [batch, num_patches, d_model]

        # Map patches back to timestep resolution
        hidden = hidden.repeat_interleave(self.patch_stride, dim=1)

        # Handle edge case: num_patches * patch_stride may not equal seq_len
        if hidden.shape[1] < seq_len:
            # Repeat last patch's representation for remaining timesteps
            remainder = seq_len - hidden.shape[1]
            last_patch = hidden[:, -1:, :].expand(-1, remainder, -1)
            hidden = torch.cat([hidden, last_patch], dim=1)
        hidden = hidden[:, :seq_len, :]  # [batch, time, d_model]

        # Per-timestep classification (head dropout applied per-timestep).
        pred = self.logit(self.head_dropout(hidden))  # [batch, time, num_classes]
        return pred
