# Label-window probe: tgbn-genre

```
tgbn-genre: unit u = day, weighting = abs
  single-offset cosine peaks at a = 5 (cos 0.695)
  label window (window search (highest cosine among windows with full row coverage)): [L + 4u, L + 7u)  cos 0.637, row coverage 1.000
  -> label(L) is available from L + 7u
  documented window: [L + 0u, L + 7u)
  - label window covers only part of the documented window (0, 6)
  - label(L) is complete only at L + 7 units; using it as an input before then is look-ahead
```

## Single offsets

Cosine between the label at L and the node's edge vector in [L + a·u, L + (a+1)·u); row coverage = share of labelled nodes with at least one edge there.

| offset a | row coverage | cos_count | cos_abs |
|---|---|---|---|
| -8 | 0.717 | 0.254 | 0.251 |
| -7 | 0.727 | 0.254 | 0.252 |
| -6 | 0.707 | 0.249 | 0.245 |
| -5 | 0.698 | 0.247 | 0.245 |
| -4 | 0.727 | 0.254 | 0.254 |
| -3 | 0.736 | 0.259 | 0.255 |
| -2 | 0.744 | 0.268 | 0.264 |
| -1 | 0.759 | 0.277 | 0.274 |
| 0 | 0.765 | 0.278 | 0.274 |
| 1 | 0.762 | 0.287 | 0.285 |
| 2 | 0.750 | 0.284 | 0.283 |
| 3 | 0.782 | 0.317 | 0.314 |
| 4 | 0.794 | 0.370 | 0.372 |
| 5 (peak) | 0.827 | 0.677 | 0.695 |
| 6 | 0.999 | 0.396 | 0.394 |
| 7 | 0.802 | 0.305 | 0.302 |
| 8 | 0.745 | 0.278 | 0.275 |

## Best contiguous windows around the peak

| window [L+a0·u, L+(a1+1)·u) | row coverage | cos_abs |
|---|---|---|
| [L+5, L+6) | 0.827 | 0.695 |
| [L+4, L+6) | 0.936 | 0.681 |
| [L+3, L+6) | 0.973 | 0.654 |
| [L+4, L+7) | 1.000 | 0.637 |
| [L+5, L+7) | 0.999 | 0.636 |
