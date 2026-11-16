import math
from collections import Counter


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(hyps, refs, max_n=4):
    num = [0] * max_n
    den = [0] * max_n
    hyp_len = 0
    ref_len = 0

    for h, r in zip(hyps, refs):
        hyp_len += len(h)
        ref_len += len(r)
        for n in range(1, max_n + 1):
            hn = _ngrams(h, n)
            rn = _ngrams(r, n)
            num[n - 1] += sum(min(c, rn[g]) for g, c in hn.items())
            den[n - 1] += max(sum(hn.values()), 0)

    scores = []
    log_sum = 0.0
    for n in range(max_n):
        p = (num[n] + 1e-9) / (den[n] + 1e-9)
        log_sum += math.log(p)
        bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
        scores.append(bp * math.exp(log_sum / (n + 1)))
    return scores


def _lcs(a, b):
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        ai = a[i - 1]
        for j in range(1, n + 1):
            cur[j] = prev[j - 1] + 1 if ai == b[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[n]


def rouge_l(hyps, refs, beta=1.2):
    total = 0.0
    for h, r in zip(hyps, refs):
        if not h or not r:
            continue
        l = _lcs(h, r)
        if l == 0:
            continue
        p = l / len(h)
        rc = l / len(r)
        total += ((1 + beta ** 2) * p * rc) / (rc + beta ** 2 * p)
    return total / max(len(hyps), 1)


def evaluate_text(hyps, refs):
    b = corpus_bleu(hyps, refs)
    return {
        "BLEU-1": b[0], "BLEU-2": b[1], "BLEU-3": b[2], "BLEU-4": b[3],
        "ROUGE-L": rouge_l(hyps, refs),
    }
