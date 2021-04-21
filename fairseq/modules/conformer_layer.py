# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
    RelPositionMultiheadAttention,
    RelativeMultiheadAttention,
    ConvolutionModule
)
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor


class ConformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8) or 8
        self.attn_type = getattr(args, "encoder_attention_type", "selfattn")
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu') or "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        if args.macaron_style:
            self.macaron_fc1 = self.build_fc1(
                self.embed_dim,
                args.encoder_ffn_embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
            self.macaron_fc2 = self.build_fc2(
                args.encoder_ffn_embed_dim,
                self.embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
            self.macaron_norm = LayerNorm(self.embed_dim)
            self.ffn_scale = 0.5
        else:
            self.macaron_fc1 = None
            self.macaron_fc2 = None
            self.macaron_norm = None
            self.ffn_scale = 1.0

        if args.use_cnn_module:
            self.conv_norm = LayerNorm(self.embed_dim)
            self.conv_module = ConvolutionModule(self.embed_dim, args.cnn_module_kernel, self.activation_fn)
            self.final_norm = LayerNorm(self.embed_dim)
        else:
            self.conv_norm = False
            self.conv_module = None
            self.final_norm = None

        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.encoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.ffn_norm = LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, args):
        if self.attn_type == "selfattn":
            attn_func = MultiheadAttention
        elif self.attn_type == "rel_selfattn":
            attn_func = RelPositionMultiheadAttention
        elif self.attn_type == "relative":
            max_relative_length = getattr(args, "max_encoder_relative_length", -1)
            if max_relative_length != -1:
                return RelativeMultiheadAttention(
                    embed_dim,
                    args.encoder_attention_heads,
                    dropout=args.attention_dropout,
                    self_attention=True,
                    q_noise=self.quant_noise,
                    qn_block_size=self.quant_noise_block_size,
                    max_relative_length=max_relative_length,
                )
            else:
                print("The maximum encoder relative length %d can not be -1!" % max_relative_length)
                exit(1)
        else:
            attn_func = MultiheadAttention
            print("The attention type %s is not supported!" % self.attn_type)
            exit(1)

        return attn_func(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x,
                encoder_padding_mask: Optional[Tensor],
                attn_mask: Optional[Tensor] = None,
                pos_emb: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.
            positions (Tensor): the position embedding for relative position encoding

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        # whether to use macaron style
        if self.macaron_norm is not None:
            residual = x
            if self.normalize_before:
                x = self.macaron_norm(x)
            x = self.macaron_fc2(self.activation_dropout_module(self.activation_fn(self.macaron_fc1(x))))
            x = residual + self.ffn_scale * self.dropout_module(x)
            if not self.normalize_before:
                x = self.macaron_norm(x)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if self.attn_type == "rel_selfattn":
            assert pos_emb is not None, "Positions is necessary for RPE!"
            x, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                need_weights=False,
                attn_mask=attn_mask,
                pos_emb=pos_emb
            )
        else:
            x, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                need_weights=False,
                attn_mask=attn_mask,
            )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # convolution module
        if self.conv_module is not None:
            x = x.transpose(0, 1)
            residual = x
            if self.normalize_before:
                x = self.conv_norm(x)
            x = residual + self.dropout_module(self.conv_module(x, encoder_padding_mask))
            if not self.normalize_before:
                x = self.conv_norm(x)
            x = x.transpose(0, 1)

        residual = x
        if self.normalize_before:
            x = self.ffn_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(self.ffn_scale * x, residual)
        if not self.normalize_before:
            x = self.ffn_norm(x)

        if self.conv_module is not None:
            x = self.final_norm(x)

        return x
