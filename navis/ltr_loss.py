"""LambdaNDCGLoss2 from pytorchltr, vendored for the NAVIS runs.

Source: https://github.com/rjagerman/pytorchltr (pytorchltr/loss/pairwise_lambda.py and
the helpers it imports). The only change is the clamp that the NAVIS instructions ask for:

    exponent = exponent.clamp(0, 30)

With this file next to navis_audit.py, pytorchltr does not need to be installed.

------------------------------------------------------------------------------------------
MIT License

Copyright (c) 2020 Rolf Jagerman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from typing import Optional

import torch as _torch


def mask_padded_values(xs, n, mask_value: float = -float("inf"), mutate: bool = False):
    mask = _torch.repeat_interleave(
        _torch.arange(xs.shape[1], device=xs.device).reshape((1, xs.shape[1])),
        xs.shape[0], dim=0)
    n_mask = _torch.repeat_interleave(n.reshape((n.shape[0], 1)), xs.shape[1], dim=1)
    if not mutate:
        xs = xs.clone()
    xs[mask >= n_mask] = mask_value
    return xs


def tiebreak_argsort(x, descending: bool = True, generator: Optional[_torch.Generator] = None):
    rng_kwargs = {"generator": generator} if generator is not None else {}
    p = _torch.randperm(x.shape[1], device=x.device, **rng_kwargs)
    return p[_torch.argsort(x[:, p], descending=descending)]


def rank_by_score(scores, n, generator: Optional[_torch.Generator] = None):
    if scores.dim() == 3:
        scores = scores.reshape((scores.shape[0], scores.shape[1]))
    return tiebreak_argsort(mask_padded_values(scores, n), generator=generator)


def batch_pairs(x):
    if x.dim() == 2:
        x = x.reshape((x.shape[0], x.shape[1], 1))
    x_ij = _torch.repeat_interleave(x, x.shape[1], dim=2)
    x_ji = _torch.repeat_interleave(x.permute(0, 2, 1), x.shape[1], dim=1)
    return _torch.stack([x_ij, x_ji], dim=3)


class LambdaLoss(_torch.nn.Module):
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def _loss_per_doc_pair(self, score_pairs, rel_pairs, n):
        raise NotImplementedError

    def _loss_reduction(self, loss_pairs):
        return loss_pairs.view(loss_pairs.shape[0], -1).sum(1)

    def forward(self, scores, relevance, n):
        if relevance.ndimension() == 2:
            relevance = relevance.reshape((relevance.shape[0], relevance.shape[1], 1))
        if scores.ndimension() == 2:
            scores = scores.reshape((scores.shape[0], scores.shape[1], 1))
        ranking = rank_by_score(scores, n)
        ranking = ranking.view((ranking.shape[0], ranking.shape[1], 1))
        scores = _torch.gather(scores, 1, ranking)
        relevance = _torch.gather(relevance, 1, ranking)
        score_pairs = batch_pairs(scores)
        rel_pairs = batch_pairs(relevance)
        loss_pairs = self._loss_per_doc_pair(score_pairs, rel_pairs, n)
        n_grid = n[:, None, None].repeat(1, score_pairs.shape[1], score_pairs.shape[2])
        arange = _torch.arange(score_pairs.shape[1], device=score_pairs.device)
        range_grid = _torch.max(*_torch.meshgrid([arange, arange]))
        range_grid = range_grid[None, :, :].repeat(n.shape[0], 1, 1)
        loss_pairs[n_grid <= range_grid] = 0.0
        return self._loss_reduction(loss_pairs)


def _max_dcg(relevance, n, exp: bool = True):
    ranking = rank_by_score(relevance.double(), n)
    arange = _torch.arange(ranking.shape[1], device=relevance.device)
    discounts = _torch.log2(2.0 + arange)
    gains = _torch.gather(relevance, 1, ranking)
    gains[n[:, None] <= arange[None, :]] = 0.0
    if exp:
        gains = (2 ** gains) - 1.0
    return _torch.sum(gains / discounts[None, :], dim=1)


def _ndcg_gains(score_pairs, rel_pairs, n, exp: bool = True):
    gains = rel_pairs[:, :, :, :]
    if exp:
        gains = (2 ** gains) - 1.0
    max_dcg = _max_dcg(rel_pairs[:, :, 0, 0], n, exp)
    max_dcg[max_dcg == 0.0] = 1.0
    return gains / max_dcg[:, None, None, None]


class LambdaNDCGLoss2(LambdaLoss):
    def _loss_per_doc_pair(self, score_pairs, rel_pairs, n):
        score_diffs = score_pairs[:, :, :, 0] - score_pairs[:, :, :, 1]
        rel_diffs = rel_pairs[:, :, :, 0] - rel_pairs[:, :, :, 1]
        gains = _ndcg_gains(score_pairs, rel_pairs, n)
        gain_diffs = gains[:, :, :, 0] - gains[:, :, :, 1]
        arange = _torch.arange(score_pairs.shape[1] + 1, device=score_pairs.device)
        discounts = _torch.log2(2.0 + arange)
        idx1 = _torch.abs(arange[:-1, None] - arange[None, :-1])
        idx2 = idx1 + 1
        delta = _torch.abs(1.0 / discounts[idx1] - 1.0 / discounts[idx2])
        exponent = delta[None, :, :] * _torch.abs(gain_diffs)
        exponent = exponent.clamp(0, 30)  # NAVIS instructions.MD
        sigmoid = (1.0 / (1.0 + _torch.exp(-self.sigma * score_diffs)))
        loss = _torch.log2(sigmoid ** exponent)
        loss[rel_diffs <= 0] = 0.0
        return -loss
