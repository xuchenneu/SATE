#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from typing import Optional, Tuple
import string

import pandas as pd
import torchaudio
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    load_df_from_tsv,
    save_df_to_tsv,
)
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import download_url, extract_archive
from tqdm import tqdm


log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]


class CoVoST(Dataset):
    """Create a Dataset for CoVoST (https://github.com/facebookresearch/covost).

    Args:
        root (str): root path to the dataset and generated manifests/features
        source_language (str): source (audio) language
        target_language (str, optional): target (text) language,
        None for no translation (default: None)
        version (int, optional): CoVoST version. (default: 2)
        download (bool, optional): Whether to download the dataset if it is not
        found at root path. (default: ``False``).
    """

    COVOST_URL_TEMPLATE = (
        "https://dl.fbaipublicfiles.com/covost/"
        "covost_v2.{src_lang}_{tgt_lang}.tsv.tar.gz"
    )

    VERSIONS = {2}
    # SPLITS = ["train", "dev", "test"]
    SPLITS = ["train"]

    XX_EN_LANGUAGES = {
        1: ["fr", "de", "nl", "ru", "es", "it", "tr", "fa", "sv-SE", "mn", "zh-CN"],
        2: [
            "fr",
            "de",
            "es",
            "ca",
            "it",
            "ru",
            "zh-CN",
            "pt",
            "fa",
            "et",
            "mn",
            "nl",
            "tr",
            "ar",
            "sv-SE",
            "lv",
            "sl",
            "ta",
            "ja",
            "id",
            "cy",
        ],
    }
    EN_XX_LANGUAGES = {
        1: [],
        2: [
            "de",
            "tr",
            "fa",
            "sv-SE",
            "mn",
            "zh-CN",
            "cy",
            "ca",
            "sl",
            "et",
            "id",
            "ar",
            "ta",
            "lv",
            "ja",
        ],
    }

    def __init__(
        self,
        root: str,
        split: str,
        source_language: str,
        target_language: Optional[str] = None,
        version: int = 2,
    ) -> None:
        assert version in self.VERSIONS and split in self.SPLITS
        assert source_language is not None
        self.no_translation = target_language is None
        if not self.no_translation:
            assert "en" in {source_language, target_language}
            if source_language == "en":
                assert target_language in self.EN_XX_LANGUAGES[version]
            else:
                assert source_language in self.XX_EN_LANGUAGES[version]
        else:
            # Hack here so that we can get "split" column from CoVoST TSV.
            # Note that we use CoVoST train split for ASR which is an extension
            # to Common Voice train split.
            target_language = "de" if source_language == "en" else "en"

        self.root: Path = Path(root)

        cv_tsv_path = self.root / "validated.tsv"
        assert cv_tsv_path.is_file()
        cv_tsv = load_df_from_tsv(cv_tsv_path)

        if self.no_translation:
            print("No target translation.")
            df = cv_tsv[["path", "sentence", "client_id"]]
            df = df.set_index(["path"], drop=False)
        else:
            covost_url = self.COVOST_URL_TEMPLATE.format(
                src_lang=source_language, tgt_lang=target_language
            )
            covost_archive = self.root / Path(covost_url).name
            if not covost_archive.is_file():
                download_url(covost_url, self.root.as_posix(), hash_value=None)
            extract_archive(covost_archive.as_posix())

            covost_tsv = load_df_from_tsv(
                self.root / Path(covost_url).name.replace(".tar.gz", "")
            )
            df = pd.merge(
                left=cv_tsv[["path", "sentence", "client_id"]],
                right=covost_tsv[["path", "translation", "split"]],
                how="inner",
                on="path",
            )
            if split == "train":
                df = df[(df["split"] == split) | (df["split"] == f"{split}_covost")]
            else:
                df = df[df["split"] == split]

        data = df.to_dict(orient="index").items()
        data = [v for k, v in sorted(data, key=lambda x: x[0])]
        self.data = []
        for e in data:
            try:
                path = self.root / "wav" / e["path"]
                _ = torchaudio.info(path.as_posix())
                self.data.append(e)
            except RuntimeError:
                pass

    def __getitem__(
        self, n: int
    ) -> Tuple[Path, int, int, str, str, str, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(wav_path, sample_rate, n_frames, sentence, translation, speaker_id,
            sample_id)``
        """
        data = self.data[n]
        path = self.root / "wav" / data["path"]
        info = torchaudio.info(path)
        sample_rate = info.sample_rate
        n_frames = info.num_frames
        sentence = data["sentence"]
        translation = None if self.no_translation else data["translation"]
        speaker_id = data["client_id"]
        _id = data["path"].replace(".mp3", "")
        return path, sample_rate, n_frames, sentence, translation, speaker_id, _id

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    root = Path(args.data_root).absolute() / args.src_lang
    output_root = Path(args.output_root).absolute()
    if args.tgt_lang is not None:
        output_root = output_root / f"{args.src_lang}-{args.tgt_lang}"
    else:
        output_root = output_root / f"{args.src_lang}"
    if not root.is_dir():
        raise NotADirectoryError(f"{root} does not exist")

    zip_path = output_root / "fbank80.zip"
    if not zip_path.exists():
        # Extract features
        feature_root = output_root / "fbank80"
        feature_root.mkdir(exist_ok=True)

        for split in CoVoST.SPLITS:
            print(f"Fetching split {split}...")
            dataset = CoVoST(root, split, args.src_lang, args.tgt_lang)
            print("Extracting log mel filter bank features...")
            for wav_path, sample_rate, _, _, _, _, utt_id in tqdm(dataset):
                waveform, sample_rate = torchaudio.load(wav_path)
                extract_fbank_features(
                    waveform, sample_rate, feature_root / f"{utt_id}.npy"
                )
        # Pack features into ZIP
        print("ZIPing features...")
        create_zip(feature_root, zip_path)

        # # Clean up
        # shutil.rmtree(feature_root)

    print("Fetching ZIP manifest...")
    zip_manifest = get_zip_manifest(zip_path)
    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    task = args.task
    # if args.tgt_lang is not None:
    #     task = f"st_{args.src_lang}_{args.tgt_lang}"
    for split in CoVoST.SPLITS:
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        if args.task == "st" and args.add_src:
            manifest["src_text"] = []
        dataset = CoVoST(root, split, args.src_lang, args.tgt_lang)
        for _, sr, n_frames, src_utt, tgt_utt, speaker_id, utt_id in tqdm(dataset):
            manifest["id"].append(utt_id)
            manifest["audio"].append(zip_manifest[utt_id])
            duration_ms = int(n_frames / sr * 1000)
            manifest["n_frames"].append(int(1 + (duration_ms - 25) / 10))
            if args.lowercase_src:
                src_utt = src_utt.lower()
            if args.rm_punc_src:
                for w in string.punctuation:
                    src_utt = src_utt.replace(w, "")
                src_utt = src_utt.replace("  ", "")
            manifest["tgt_text"].append(src_utt if args.tgt_lang is None else tgt_utt)
            if args.task == "st" and args.add_src:
                manifest["src_text"].append(src_utt)
            manifest["speaker"].append(speaker_id)
        is_train_split = split.startswith("train")
        if is_train_split:
            if args.task == "st" and args.add_src and args.share:
                train_text.extend(manifest["src_text"])
            train_text.extend(manifest["tgt_text"])
        df = pd.DataFrame.from_dict(manifest)
        df = filter_manifest_df(df, is_train_split=is_train_split)
        save_df_to_tsv(df, output_root / f"{split}_{task}.tsv")

    # Generate vocab
    v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{v_size_str}_{task}"
    asr_spm_filename = None
    gen_vocab_flag = True

    if args.task == "st" and args.add_src:
        if args.share:
            if args.st_spm_prefix is not None:
                gen_vocab_flag = False
                spm_filename_prefix = args.st_spm_prefix
            else:
                spm_filename_prefix = f"spm_{args.vocab_type}{v_size_str}_{args.task}_share"
            asr_spm_filename = spm_filename_prefix + ".model"
        else:
            if args.st_spm_prefix is not None:
                gen_vocab_flag = False
                spm_filename_prefix = args.st_spm_prefix
            assert args.asr_prefix is not None
            asr_spm_filename = args.asr_prefix + ".model"
    elif args.task == "asr":
        if args.asr_prefix is not None:
            gen_vocab_flag = False
            spm_filename_prefix = args.asr_prefix

    if gen_vocab_flag:
        with NamedTemporaryFile(mode="w") as f:
            for t in train_text:
                f.write(t + "\n")
            gen_vocab(
                Path(f.name),
                output_root / spm_filename_prefix,
                args.vocab_type,
                args.vocab_size
            )
    # Generate config YAML
    gen_config_yaml(
        output_root,
        spm_filename_prefix + ".model",
        yaml_filename=f"config_{task}.yaml",
        specaugment_policy="lb",
        cmvn_type=args.cmvn_type,
        asr_spm_filename=asr_spm_filename,
        share_src_and_tgt=True if args.task == "asr" else False
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", "-d", required=True, type=str,
        help="data root with sub-folders for each language <root>/<src_lang>"
    )
    parser.add_argument(
        "--output-root", "-o", required=True, type=str,
        help="output root to save the results"
    )
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=1000, type=int)
    parser.add_argument("--src-lang", "-s", required=True, type=str)
    parser.add_argument("--task", type=str, default="asr", choices=["asr", "st"])
    parser.add_argument("--tgt-lang", "-t", type=str)
    parser.add_argument("--share", action="store_true",
                        help="share the tokenizer and dictionary of the transcription and translation")
    parser.add_argument("--add-src", action="store_true", help="add the src text for st task")
    parser.add_argument("--asr-prefix", type=str, help="prefix of the asr dict")
    parser.add_argument("--st-spm-prefix", type=str, default=None, help="prefix of the existing st dict")
    parser.add_argument("--lowercase-src", action="store_true", help="lowercase the source text")
    parser.add_argument("--rm-punc-src", action="store_true", help="remove the punctuation of the source text")
    parser.add_argument("--cmvn-type", default="utterance",
                        choices=["global", "utterance"],
                        help="The type of cepstral mean and variance normalization")
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
