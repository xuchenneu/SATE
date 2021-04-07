# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.data.data_utils import post_process

from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion


@register_criterion("label_smoothed_cross_entropy_with_ctc")
class LabelSmoothedCrossEntropyCriterionWithCTC(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self, task, sentence_avg, label_smoothing, post_process="letter", ctc_weight=0.0):
        super().__init__(task, sentence_avg, label_smoothing)
        self.blank_idx = task.target_dictionary.index(task.blank_symbol) if hasattr(task, 'blank_symbol') else 0
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.report_accuracy = True

        assert 0 <= ctc_weight <= 1
        self.ctc_weight = ctc_weight
        if self.ctc_weight > 0:
            assert getattr(task, "src_dict", None) is not None, "CTC need a source dictionary."
            self.zero_infinity = True
            self.post_process = post_process

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument(
            "--zero-infinity",
            default=True,
            type=bool,
            help="zero inf loss when source length <= target length",
        )
        parser.add_argument(
            "--ctc-weight",
            default=0.0,
            type=float,
            metavar="D",
            help="weight of CTC loss",
        )
        parser.add_argument(
            "--post-process",
            default="letter",
            type=str,
            help="how to post process predictions into words. can be letter, "
            "wordpiece, BPE symbols, etc. "
            "See fairseq.data.data_utils.post_process() for full list of options",
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        src_tokens, src_lengths, prev_output_tokens = sample["net_input"].values()
        encoder_out = model.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        net_output = model.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )

        # net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "trans_loss": utils.item(loss.data) if reduce else loss.data,
            "nll_loss": utils.item(nll_loss.data) if reduce else nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        if self.ctc_weight > 0:
            ctc_loss = self.compute_ctc_loss(model, sample, encoder_out)
            logging_output["ctc_loss"] = utils.item(ctc_loss.data)
            loss = (1 - self.ctc_weight) * loss + self.ctc_weight * ctc_loss
            logging_output["loss"] = utils.item(loss.data) if reduce else loss.data

        return loss, sample_size, logging_output

    def compute_ctc_loss(self, model, sample, encoder_out):
        transcript = sample["transcript"]
        if "ctc_logit" in encoder_out:
            ctc_logit = encoder_out["ctc_logit"][0]
        else:
            ctc_logit = model.encoder.compute_ctc_logit(encoder_out)
        lprobs = model.get_normalized_probs(
            [ctc_logit], log_probs=True
        ).contiguous()  # (T, B, C) from the encoder
        non_padding_mask = ~encoder_out["encoder_padding_mask"][0]
        input_lengths = non_padding_mask.long().sum(-1)

        pad_mask = (transcript["tokens"] != self.pad_idx) & (
                transcript["tokens"] != self.eos_idx
        )
        targets_flat = transcript["tokens"].masked_select(pad_mask)
        transcript_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                transcript_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

        logging_output = {
            "ctc_loss": utils.item(loss.data),  # * sample['ntokens'],
        }

        if not model.training:
            import editdistance

            with torch.no_grad():
                lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()

                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0
                for lp, t, inp_l in zip(
                    lprobs_t,
                    sample["target_label"]
                    if "target_label" in sample
                    else sample["target"],
                    input_lengths,
                ):
                    lp = lp[:inp_l].unsqueeze(0)

                    decoded = None
                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_units_arr = targ.tolist()

                    toks = lp.argmax(dim=-1).unique_consecutive()
                    pred_units_arr = toks[toks != self.blank_idx].tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()

                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    if decoded is not None and "words" in decoded:
                        pred_words = decoded["words"]
                        w_errs += editdistance.eval(pred_words, targ_words)
                        wv_errs += editdistance.eval(pred_words_raw, targ_words)
                    else:
                        dist = editdistance.eval(pred_words_raw, targ_words)
                        w_errs += dist
                        wv_errs += dist

                    w_len += len(targ_words)

                logging_output["wv_errors"] = wv_errs
                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        trans_loss_sum = utils.item(
            sum(log.get("trans_loss", 0) for log in logging_outputs)
        )
        nll_loss_sum = utils.item(
            sum(log.get("nll_loss", 0) for log in logging_outputs)
        )
        ctc_loss_sum = utils.item(
            sum(log.get("ctc_loss", 0) for log in logging_outputs)
        )
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "trans_loss", trans_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "ctc_loss",
            ctc_loss_sum / sample_size / math.log(2),
            sample_size,
            round=3,
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
