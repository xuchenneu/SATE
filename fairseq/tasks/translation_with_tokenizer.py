# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import itertools
import json
import logging
import os
import os.path as op
from typing import Optional
from argparse import Namespace
from omegaconf import II
import csv
from typing import Dict, List, Optional, Tuple

import numpy as np
from fairseq import metrics, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)
from fairseq.data.fairseq_dataset import FairseqDataset
from functools import lru_cache
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, TranslationConfig


EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


class ListTextDataset(FairseqDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, text, dictionary, append_eos=True, reverse_order=False):
        self.tokens_list = []
        self.lines = text
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.read_data(dictionary)
        self.size = len(self.tokens_list)

    def read_data(self, dictionary):
        for line in self.lines:
            tokens = dictionary.encode_line(
                line,
                add_if_not_exist=False,
                append_eos=self.append_eos,
                reverse_order=self.reverse_order,
            ).long()
            self.tokens_list.append(tokens)
            self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]


def load_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    pre_tokenizer=None,
    src_bpe_tokenizer=None,
    tgt_bpe_tokenizer=None,
):
    def tokenize_text(text: str, pre_tokenizer, bpe_tokenizer):
        if pre_tokenizer is not None:
            text = pre_tokenizer.encode(text)
        if bpe_tokenizer is not None:
            text = bpe_tokenizer.encode(text)
        return text

    src_datasets = []
    tgt_datasets = []
    root = data_path
    _splits = split.split(",")
    for split in _splits:
        src_texts = []
        tgt_texts = []
        tsv_path = op.join(root, f"{split}.tsv")
        if not op.isfile(tsv_path):
            raise FileNotFoundError(f"Dataset not found: {tsv_path}")
        with open(tsv_path) as f:
            reader = csv.DictReader(
                f,
                delimiter="\t",
                quotechar=None,
                doublequote=False,
                lineterminator="\n",
                quoting=csv.QUOTE_NONE,
            )

            for e in reader:
                item = dict(e)
                src_text = item.get("src_text", "")
                tgt_text = item.get("tgt_text", "")

                src_tokenized = tokenize_text(src_text, pre_tokenizer, src_bpe_tokenizer)
                tgt_tokenized = tokenize_text(tgt_text, pre_tokenizer, tgt_bpe_tokenizer)

                src_texts.append(src_tokenized)
                tgt_texts.append(tgt_tokenized)

            assert len(src_texts) > 0 and len(src_texts) == len(tgt_texts), (len(src_texts), len(tgt_texts))

            src_dataset = ListTextDataset(src_texts, dictionary=src_dict)
            if truncate_source:
                src_dataset = AppendTokenDataset(
                    TruncateDataset(
                        StripTokenDataset(src_dataset, src_dict.eos()),
                        max_source_positions - 1,
                    ),
                    src_dict.eos(),
                )
            src_datasets.append(src_dataset)

            tgt_dataset = ListTextDataset(tgt_texts, dictionary=tgt_dict)
            if tgt_dataset is not None:
                tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split, src, tgt, len(src_datasets[-1])
            )
        )

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )


class TransDataConfig(object):
    """Wrapper class for data config YAML"""

    def __init__(self, yaml_path):
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load YAML files for " "S2T data config")
        self.config = {}
        if op.isfile(yaml_path):
            try:
                with open(yaml_path) as f:
                    self.config = yaml.load(f, Loader=yaml.FullLoader)
            except Exception as e:
                logger.info(f"Failed to load config from {yaml_path}: {e}")
        else:
            logger.error(f"Cannot find {yaml_path}")

    @property
    def src_vocab_filename(self):
        """fairseq vocabulary file under data root"""
        return self.config.get("src_vocab_filename", None)

    @property
    def tgt_vocab_filename(self):
        """fairseq vocabulary file under data root"""
        return self.config.get("tgt_vocab_filename", None)

    @property
    def share_src_and_tgt(self) -> bool:
        return self.config.get("share_src_and_tgt", False)

    @property
    def shuffle(self) -> bool:
        """Shuffle dataset samples before batching"""
        return self.config.get("shuffle", False)

    @property
    def pre_tokenizer(self) -> Dict:
        """Pre-tokenizer to apply before subword tokenization. Returning
        a dictionary with `tokenizer` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        return self.config.get("pre_tokenizer", {"tokenizer": None})

    @property
    def src_bpe_tokenizer(self) -> Dict:
        """Subword tokenizer to apply after pre-tokenization. Returning
        a dictionary with `bpe` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        return self.config.get("src_bpe_tokenizer", {"bpe": None})

    @property
    def tgt_bpe_tokenizer(self) -> Dict:
        """Subword tokenizer to apply after pre-tokenization. Returning
        a dictionary with `bpe` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        return self.config.get("tgt_bpe_tokenizer", None)

    @property
    def prepend_tgt_lang_tag(self) -> bool:
        """Prepend target lang ID token as the target BOS (e.g. for to-many
        multilingual setting). During inference, this requires `--prefix-size 1`
        to force BOS to be lang ID token."""
        return self.config.get("prepend_tgt_lang_tag", False)


@dataclass
class TranslationWithTokenizerConfig(TranslationConfig):
    config_yaml: Optional[str] = field(
        default="config.yaml",
        metadata={
            "help": "Configuration YAML filename (under manifest root)"
        },
    )


@register_task("translation_with_tokenizer", dataclass=TranslationWithTokenizerConfig)
class TranslationWithTokenizerTask(TranslationTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: TranslationWithTokenizerConfig

    def __init__(self, cfg: TranslationWithTokenizerConfig, src_dict, tgt_dict, data_cfg,
                 pre_tokenizer, src_bpe_tokenizer, tgt_bpe_tokenizer):
        super().__init__(cfg, src_dict, tgt_dict)
        self.data_cfg = data_cfg
        self.pre_tokenizer = pre_tokenizer
        self.src_bpe_tokenizer = src_bpe_tokenizer
        self.tgt_bpe_tokenizer = tgt_bpe_tokenizer

    @classmethod
    def setup_task(cls, cfg: TranslationWithTokenizerConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (argparse.Namespace): parsed command-line arguments
        """
        data_cfg = TransDataConfig(op.join(cfg.data, cfg.config_yaml))

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load
        src_dict_path = op.join(cfg.data, data_cfg.src_vocab_filename)
        if not op.isfile(src_dict_path):
            raise FileNotFoundError(f"source Dict not found: {src_dict_path}")
        src_dict = cls.load_dictionary(src_dict_path)

        tgt_dict_path = op.join(cfg.data, data_cfg.tgt_vocab_filename)
        if not op.isfile(tgt_dict_path):
            raise FileNotFoundError(f"target Dict not found: {src_dict_path}")
        tgt_dict = cls.load_dictionary(src_dict_path)

        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] source dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] target dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        logger.info(f"pre-tokenizer: {data_cfg.pre_tokenizer}")
        pre_tokenizer = encoders.build_tokenizer(Namespace(**data_cfg.pre_tokenizer))

        logger.info(f"source tokenizer: {data_cfg.src_bpe_tokenizer}")
        src_bpe_tokenizer = encoders.build_bpe(Namespace(**data_cfg.src_bpe_tokenizer))
        logger.info(f"target tokenizer: {data_cfg.tgt_bpe_tokenizer}")
        tgt_bpe_tokenizer = encoders.build_bpe(Namespace(**data_cfg.tgt_bpe_tokenizer))

        return cls(cfg, src_dict, tgt_dict, data_cfg, pre_tokenizer, src_bpe_tokenizer, tgt_bpe_tokenizer)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            load_alignments=self.cfg.load_alignments,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
            pre_tokenizer=self.pre_tokenizer,
            src_bpe_tokenizer=self.src_bpe_tokenizer,
            tgt_bpe_tokenizer=self.tgt_bpe_tokenizer,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        if self.cfg.eval_bleu:
            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args)
            )

            gen_args = json.loads(self.cfg.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def build_tokenizer(self, args):
        logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))

    def build_src_bpe(self):
        logger.info(f"source tokenizer: {self.data_cfg.src_bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.src_bpe_tokenizer))

    def build_src_bpe(self, args):
        logger.info(f"src tokenizer: {self.data_cfg.src_bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.src_bpe_tokenizer))

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.cfg.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])
