#!/usr/bin/env python3

import logging

import torch.nn as nn
from fairseq import checkpoint_utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text import S2TTransformerModel, S2TTransformerEncoder
from fairseq.modules import (
    ConformerEncoderLayer,
)

logger = logging.getLogger(__name__)


@register_model("s2t_conformer")
class S2TConformerModel(S2TTransformerModel):
    """Adapted Transformer model (https://arxiv.org/abs/1706.03762) for
    speech-to-text tasks. The Transformer encoder/decoder remains the same.
    A trainable input subsampler is prepended to the Transformer encoder to
    project inputs into the encoder dimension as well as downsample input
    sequence for computational efficiency."""

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        S2TTransformerModel.add_args(parser)

        parser.add_argument(
            "--macaron-style",
            default=False,
            type=bool,
            help="Whether to use macaron style for positionwise layer",
        )
        # Attention
        parser.add_argument(
            "--zero-triu",
            default=False,
            type=bool,
            help="If true, zero the uppper triangular part of attention matrix.",
        )
        # Relative positional encoding
        parser.add_argument(
            "--rel-pos-type",
            type=str,
            default="legacy",
            choices=["legacy", "latest"],
            help="Whether to use the latest relative positional encoding or the legacy one."
                 "The legacy relative positional encoding will be deprecated in the future."
                 "More Details can be found in https://github.com/espnet/espnet/pull/2816.",
        )
        # CNN module
        parser.add_argument(
            "--use-cnn-module",
            default=False,
            type=bool,
            help="Use convolution module or not",
        )
        parser.add_argument(
            "--cnn-module-kernel",
            default=31,
            type=int,
            help="Kernel size of convolution module.",
        )
        pass

    @classmethod
    def build_encoder(cls, args, task=None, embed_tokens=None):
        encoder = S2TConformerEncoder(args, task, embed_tokens)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from, strict=False
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
        return encoder


class S2TConformerEncoder(S2TTransformerEncoder):
    """Speech-to-text Conformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args, task=None, embed_tokens=None):
        super().__init__(args, task, embed_tokens)

        del self.layers

        self.layers = nn.ModuleList(
            [ConformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        )

    def forward(self, src_tokens, src_lengths):
        if self.history is not None:
            self.history.clean()

        x, input_lengths = self.subsample(src_tokens, src_lengths)
        x = self.embed_scale * x

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        if self.attn_type != "rel_selfattn":
            x += positions
        x = self.dropout_module(x)
        positions = self.dropout_module(positions)

        # add emb into history
        if self.history is not None:
            self.history.add(x)

        for layer in self.layers:
            if self.history is not None:
                x = self.history.pop()
            x = layer(x, encoder_padding_mask, pos_emb=positions)
            if self.history is not None:
                self.history.add(x)

        if self.history is not None:
            x = self.history.pop()

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }


@register_model_architecture(model_name="s2t_conformer", arch_name="s2t_conformer")
def base_architecture(args):
    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # Conformer
    args.macaron_style = getattr(args, "macaron_style", True)
    args.use_cnn_module = getattr(args, "use_cnn_module", True)
    args.cnn_module_kernel = getattr(args, "cnn_module_kernel", 31)

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_type = getattr(args, "encoder_attention_type", "selfattn")
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_type = getattr(args, "decoder_attention_type", "selfattn")
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    args.max_encoder_relative_length = getattr(args, 'max_encoder_relative_length', -1)
    args.max_decoder_relative_length = getattr(args, 'max_decoder_relative_length', -1)
    args.k_only = getattr(args, 'k_only', True)


@register_model_architecture("s2t_conformer", "s2t_conformer_s")
def s2t_conformer_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)


@register_model_architecture("s2t_conformer", "s2t_conformer_s_relative")
def s2t_conformer_s_relative(args):
    args.max_encoder_relative_length = 100
    args.max_decoder_relative_length = 20
    args.k_only = True
    s2t_conformer_s(args)


@register_model_architecture("s2t_conformer", "s2t_conformer_xs")
def s2t_conformer_xs(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 4)
    args.dropout = getattr(args, "dropout", 0.3)
    s2t_conformer_s(args)


@register_model_architecture("s2t_conformer", "s2t_conformer_sp")
def s2t_conformer_sp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_conformer_s(args)


@register_model_architecture("s2t_conformer", "s2t_conformer_m")
def s2t_conformer_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    base_architecture(args)


@register_model_architecture("s2t_conformer", "s2t_conformer_mp")
def s2t_conformer_mp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_conformer_m(args)


@register_model_architecture("s2t_conformer", "s2t_conformer_l")
def s2t_conformer_l(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.2)
    base_architecture(args)


@register_model_architecture("s2t_conformer", "s2t_conformer_lp")
def s2t_conformer_lp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_conformer_l(args)
