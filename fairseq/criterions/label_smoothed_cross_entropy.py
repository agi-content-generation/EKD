# # Copyright (c) Facebook, Inc. and its affiliates.
# #
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F # sherwin
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion("label_smoothed_cross_entropy")
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        learn_from_teacher_model=False,
        teacher_model_path=None,
        distil_strategy='normal',  
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

        self.range_eps = 0.01
        self.queue = torch.FloatTensor([])
        self.teacher_loss_queue = torch.FloatTensor([])
        self.real_distil_rate = 0.5  
        self.dict_count = None

        self.learn_from_teacher_model = learn_from_teacher_model
        self.teacher_model = None
        self.distil_strategy = distil_strategy  

        if self.learn_from_teacher_model:
            from fairseq.checkpoint_utils import load_model_ensemble
            if teacher_model_path:
                self.teacher_model, self._model_args = load_model_ensemble(
                    utils.split_paths(teacher_model_path),
                    arg_overrides={},
                    task=task,
                    suffix=""
                )
            else:
                raise ValueError("please add teacher_model_path")

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--report-accuracy', action='store_true',
                            help='report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        parser.add_argument('--learn-from-teacher-model', action='store_true', default=False,
                            help='Whether to learn from a larger teacher model')
        parser.add_argument('--teacher-model-path', type=str, default=None,
                            help='Path to the teacher model checkpoint')
        parser.add_argument('--distil-strategy', type=str, default='normal',  
                            help='Distillation strategy (default: normal)')


    
    def forward(self, model, sample, reduce=True, teacher_model=None, update_num=None):
        """
        Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if self.teacher_model:
            net_output = model(**sample['net_input'])
            teacher_output = None
            teacher_model = self.teacher_model[0] 
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
            teacher_model = teacher_model.to(device)
            
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_output = teacher_model(**sample['net_input'])  
    
            loss, nll_loss, extra_result = self.compute_loss(model, net_output, sample, reduce=reduce,
                                                             teacher_output=teacher_output,
                                                             distil_strategy=self.distil_strategy,
                                                             update_num=update_num)
            sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']  # todo:sherwin;
            logging_output = {
                'loss': loss.data,
                'nll_loss': nll_loss.data if nll_loss is not None else loss.data,
                'ntokens': sample['ntokens'],
                'nsentences': sample['target'].size(0),
                'sample_size': sample_size,
                'distil_rate': self.real_distil_rate,
                'gpu_nums': 1,
                'KD_loss': extra_result['KD_loss'].data if extra_result.get('KD_loss', None) is not None else 0,
                'nll_loss_distil': extra_result['nll_loss_distil'].data if extra_result.get('nll_loss_distil',
                                                                                            None) is not None else 0,
            }
            return loss, sample_size, logging_output
            
        else:
            net_output = model(**sample["net_input"])
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce, teacher_output=None, distil_strategy="normal", update_num=None)
            sample_size = (sample["target"].size(0) if self.sentence_avg else sample["ntokens"])
            logging_output = {
                "loss": loss.data,
                "nll_loss": nll_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
                'distil_rate': self.real_distil_rate,
                'gpu_nums': 1,
                'KD_loss': extra_result['KD_loss'].data if extra_result.get('KD_loss', None) is not None else 0,
                'nll_loss_distil': extra_result['nll_loss_distil'].data if extra_result.get('nll_loss_distil',
                                                                                            None) is not None else 0,
            }
            if self.report_accuracy:
                n_correct, total = self.compute_accuracy(model, net_output, sample)
                logging_output["n_correct"] = utils.item(n_correct.data)
                logging_output["total"] = utils.item(total.data)
            return loss, sample_size, logging_output
        

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def get_teacher_probs(self, teacher_output):
        teacher_predict = teacher_output[0]
        teacher_predict = teacher_predict.view(-1, teacher_predict.size(-1)) # B*T x vocab
        self.task.args.teacher_predict_temperature_schedule = 'binary' # sherwin
        self.task.args.teacher_predict_temperature = 1.0  # sherwin

        if self.task.args.teacher_predict_temperature_schedule == 'binary':
            teacher_predict_max = torch.max(teacher_predict, dim=-1)[0].view(-1, 1) # B*T x 1
            teacher_predict_mask = teacher_predict_max > 0.5 # B*T x 1
            temperature = torch.ones_like(teacher_predict_max) / self.task.args.teacher_predict_temperature # B*T x 1
            temperature = temperature.masked_fill(teacher_predict_mask, self.task.args.teacher_predict_temperature) # B*T x 1
            teacher_predict = teacher_predict * temperature
        elif self.task.args.teacher_predict_temperature_schedule == 'topk':
            distil_lprobs = F.softmax(teacher_predict, dim=-1, dtype=torch.float32) # B * T x vocab
            distil_mask = distil_lprobs > 0.01
            invalid_mask = distil_mask.sum(dim=-1) == 0
            distil_mask[invalid_mask, :] = True
            teacher_predict.masked_fill_(~distil_mask, float("-inf"))
        else:
            teacher_predict = teacher_predict * self.task.args.teacher_predict_temperature
        distil_lprobs = F.softmax(teacher_predict, dim=-1, dtype=torch.float32) # B x T x vocab
        return distil_lprobs

    def compute_loss(self, model, net_output, sample, reduce=True, teacher_output=None, distil_strategy="normal", update_num=None):
        probs = model.get_normalized_probs(net_output, log_probs=False)
        lprobs = torch.log(probs)
        probs = probs.view(-1, lprobs.size(-1))
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output)
        bsz, seq_len = target.shape
        target = target.view(-1, 1)
        pad_mask = target.eq(self.padding_idx).view(-1)
        loss = None
        nll_loss = None
        extra_result = {}
        if distil_strategy == 'normal' or teacher_output is None:
            result = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
            )
            loss, nll_loss = result
            
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
            )
        elif distil_strategy == 'distil_all':
            golden_loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
            )
            distil_lprobs = self.get_teacher_probs(teacher_output)
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none')
            KL_loss = KL_loss.sum(dim=-1)
            KL_loss.masked_fill_(pad_mask, 0.)
            KL_loss = KL_loss.sum()
            extra_result['KD_loss'] = KL_loss
            alpha = 0.5
            loss = alpha * golden_loss + (1 - alpha) * KL_loss

        elif distil_strategy == 'reverse_kl':
            golden_loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
            )
            
            distil_lprobs = self.get_teacher_probs(teacher_output)
            reverse_KL_loss = F.kl_div(distil_lprobs, probs, reduction='none')  
            reverse_KL_loss = reverse_KL_loss.sum(dim=-1)
            reverse_KL_loss.masked_fill_(pad_mask, 0.0)
            reverse_KL_loss = reverse_KL_loss.sum()
            
            extra_result['KD_loss'] = reverse_KL_loss
            
            alpha = 0.5  
            loss = alpha * golden_loss + (1 - alpha) * reverse_KL_loss

        return loss, nll_loss, extra_result


    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
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
