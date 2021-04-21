#!/usr/bin/env python3

import logging
import math

import torch
import torch.nn as nn
from fairseq import checkpoint_utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import Embedding, TransformerDecoder
from fairseq.models.speech_to_text import (
    S2TTransformerModel,
    S2TTransformerEncoder,
    S2TConformerEncoder,
    S2TConformerModel
)
from fairseq.models.speech_to_text.s2t_transformer import Conv1dSubsampler
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
    LearnableDenseLayerHistory
)

logger = logging.getLogger(__name__)


@register_model("s2t_sate")
class S2TSATEModel(S2TTransformerModel):
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
        S2TConformerModel.add_args(parser)

        parser.add_argument(
            "--text-encoder-layers",
            default=6,
            type=int,
            help="layers of the text encoder",
        )
        parser.add_argument(
            "--adapter",
            default="league",
            type=str,
            help="adapter type",
        )
        parser.add_argument(
            "--acoustic-encoder",
            default="transformer",
            type=str,
            help="the architecture of the acoustic encoder",
        )
        parser.add_argument(
            "--load-pretrained-acoustic-encoder-from",
            type=str,
            metavar="STR",
            help="model to take acoustic encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--load-pretrained-text-encoder-from",
            type=str,
            metavar="STR",
            help="model to take text encoder weights from (for initialization)",
        )
        pass

    @classmethod
    def build_encoder(cls, args, task=None, embed_tokens=None):
        encoder = S2TSATEEncoder(args, task, embed_tokens)

        if getattr(args, "load_pretrained_encoder_from", None):
            logger.info(
                f"loaded pretrained acoustic encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from, strict=False
            )

        if getattr(args, "load_pretrained_acoustic_encoder_from", None):
            logger.info(
                f"loaded pretrained acoustic encoder from: "
                f"{args.load_pretrained_acoustic_encoder_from}"
            )
            encoder.acoustic_encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder.acoustic_encoder, checkpoint=args.load_pretrained_acoustic_encoder_from, strict=False
            )

        if getattr(args, "load_pretrained_text_encoder_from", None):
            logger.info(
                f"loaded pretrained text encoder from: "
                f"{args.load_pretrained_text_encoder_from}"
            )
            encoder.text_encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder.text_encoder, checkpoint=args.load_pretrained_text_encoder_from, strict=False
            )

        return encoder


class Adapter(nn.Module):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__()

        attention_dim = args.encoder_embed_dim
        self.embed_scale = math.sqrt(attention_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = dictionary.pad_index

        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )

        adapter_type = getattr(args, "adapter", "league")
        self.adapter_type = adapter_type

        if adapter_type in ["linear", "league"]:
            self.linear_adapter = nn.Sequential(
                nn.Linear(attention_dim, attention_dim),
                LayerNorm(args.encoder_embed_dim),
                self.dropout_module,
                nn.ReLU(),
            )
        elif adapter_type == "linear2":
            self.linear_adapter = nn.Sequential(
                nn.Linear(attention_dim, attention_dim),
                self.dropout_module,
            )
        elif adapter_type == "subsample":
            self.subsample_adaptor = Conv1dSubsampler(
                attention_dim,
                args.conv_channels,
                attention_dim,
                [int(k) for k in args.conv_kernel_sizes.split(",")],
            )

        if adapter_type in ["embed", "context", "league", "gated_league"]:
            if embed_tokens is None:
                num_embeddings = len(dictionary)
                self.embed_adapter = Embedding(num_embeddings, attention_dim, self.padding_idx)
            else:
                self.embed_adapter = embed_tokens

        if adapter_type == "gated_league":
            self.gate_linear = nn.Linear(2 * attention_dim, attention_dim)
        elif adapter_type == "gated_league2":
            self.gate_linear1 = nn.Linear(attention_dim, attention_dim)
            self.gate_linear2 = nn.Linear(attention_dim, attention_dim)

        attn_type = getattr(args, "text_encoder_attention_type", "selfattn")
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, args.encoder_embed_dim, self.padding_idx, pos_emb_type=attn_type
        )

    def forward(self, x, padding):

        representation, distribution = x
        batch, seq_len, embed_dim = distribution.size()
        lengths = (~padding).long().sum(-1)

        if self.adapter_type == "linear":
            out = self.linear_adapter(representation)

        elif self.adapter_type == "context":
            out = torch.mm(distribution.view(-1, embed_dim), self.embed_adapter.weight).view(batch, seq_len, -1)

        elif self.adapter_type == "subsample":
            out = self.subsample_adaptor(x, lengths)

        elif self.adapter_type == "league":
            linear_out = self.linear_adapter(representation)
            soft_out = torch.mm(distribution.view(-1, embed_dim), self.embed_adapter.weight).view(batch, seq_len, -1)
            out = linear_out + soft_out
        elif self.adapter_type == "gated_league":
            linear_out = self.linear_adapter(representation)
            soft_out = self.embed_adapter(distribution)
            coef = (self.gate_linear(torch.cat([linear_out, soft_out], dim=-1))).sigmoid()
            out = coef * linear_out + (1 - coef) * soft_out
        else:
            out = None
            logging.error("Unsupported adapter type: {}.".format(self.adapter_type))

        out = self.embed_scale * out

        positions = self.embed_positions(padding).transpose(0, 1)
        out = positions + out

        out = self.dropout_module(out)

        return out, positions


class TextEncoder(FairseqEncoder):
    def __init__(self, args, embed_tokens=None):

        super().__init__(None)

        self.embed_tokens = embed_tokens

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(args.text_encoder_layers)]
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

    def forward(self, x, encoder_padding_mask=None, positions=None, history=None):

        for layer in self.layers:
            if history is not None:
                x = history.pop()
            x = layer(x, encoder_padding_mask, pos_emb=positions)
            if history is not None:
                history.add(x)

        if history is not None:
            x = history.pop()

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return x


class S2TSATEEncoder(FairseqEncoder):
    """Speech-to-text Conformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args, task=None, embed_tokens=None):
        super().__init__(None)

        # acoustic encoder
        acoustic_encoder_type = getattr(args, "acoustic_encoder", "transformer")
        if acoustic_encoder_type == "transformer":
            self.acoustic_encoder = S2TTransformerEncoder(args, task, embed_tokens)
        elif acoustic_encoder_type == "conformer":
            self.acoustic_encoder = S2TConformerEncoder(args, task, embed_tokens)
        else:
            logging.error("Unsupported model arch {}!".format(acoustic_encoder_type))

        # adapter
        self.adapter = Adapter(args, task.source_dictionary, embed_tokens)

        # self.length_adapter = Conv1dSubsampler(
        #     args.encoder_embed_dim,
        #     args.conv_channels,
        #     args.encoder_embed_dim,
        #     [int(k) for k in args.conv_kernel_sizes.split(",")],
        # )

        # acoustic_encoder_attention_type = args.encoder_attention_type
        # args.encoder_attention_type = "selfattn"

        # text encoder
        self.text_encoder = TextEncoder(args, embed_tokens)
        # args.encoder_attention_type = acoustic_encoder_attention_type

        if getattr(args, "use_enc_dlcl", False):
            normalize_before = args.encoder_normalize_before
            layer_num = args.encoder_layers + args.text_encoder_layers + 1
            self.history = LearnableDenseLayerHistory(normalize_before, layer_num, args.encoder_embed_dim, True)
        else:
            self.history = None

    def forward(self, src_tokens, src_lengths):
        if self.history is not None:
            self.history.clean()

        acoustic_encoder_out = self.acoustic_encoder(src_tokens, src_lengths)

        encoder_out = acoustic_encoder_out["encoder_out"][0]
        encoder_padding_mask = acoustic_encoder_out["encoder_padding_mask"][0]

        ctc_logit = self.acoustic_encoder.compute_ctc_logit(encoder_out)
        ctc_prob = self.acoustic_encoder.compute_ctc_prob(encoder_out)
        x = (encoder_out, ctc_prob)

        x, positions = self.adapter(x, encoder_padding_mask)

        if self.history is not None:
            acoustic_history = self.acoustic_encoder.history
            layer_num = acoustic_history.layer_num
            idx = torch.arange(layer_num).unsqueeze(0).T.repeat(1, layer_num).to(x.device)
            self.history.weight.scatter(0, idx, acoustic_history.weight)
            self.history.layers.extend(acoustic_history.layers)
            self.history.count = acoustic_history.count
            self.history.sum = acoustic_history.sum

            self.history.add(x)

        # src_lengths = (~encoder_padding_mask).sum(1)
        # x = x.transpose(0, 1)
        # x, input_lengths = self.length_adapter(x, src_lengths)
        # encoder_padding_mask = lengths_to_padding_mask(input_lengths)

        x = self.text_encoder(x, encoder_padding_mask, positions, self.history)

        return {
            "ctc_logit": [ctc_logit],    # T x B x C
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            [] if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            [] if len(encoder_out["encoder_padding_mask"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_padding_mask"]]
        )

        new_encoder_embedding = (
            [] if len(encoder_out["encoder_embedding"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }


@register_model_architecture(model_name="s2t_sate", arch_name="s2t_sate")
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
    args.encoder_normalize_before = getattr(args, "acoustic_encoder", "transformer")
    args.encoder_normalize_before = getattr(args, "adapter", "league")
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 6)
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


@register_model_architecture("s2t_sate", "s2t_sate_s")
def s2t_sate_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)


@register_model_architecture("s2t_sate", "s2t_sate_s_relative")
def s2t_sate_s_relative(args):
    args.max_encoder_relative_length = 100
    args.max_decoder_relative_length = 20
    args.k_only = True
    s2t_sate_s(args)


@register_model_architecture("s2t_sate", "s2t_sate_xs")
def s2t_sate_xs(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 4)
    args.dropout = getattr(args, "dropout", 0.3)
    s2t_sate_s(args)


@register_model_architecture("s2t_sate", "s2t_sate_sp")
def s2t_sate_sp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_sate_s(args)


@register_model_architecture("s2t_sate", "s2t_sate_m")
def s2t_sate_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    base_architecture(args)


@register_model_architecture("s2t_sate", "s2t_sate_mp")
def s2t_sate_mp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_sate_m(args)


@register_model_architecture("s2t_sate", "s2t_sate_l")
def s2t_sate_l(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.2)
    base_architecture(args)


@register_model_architecture("s2t_sate", "s2t_sate_lp")
def s2t_sate_lp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_sate_l(args)
