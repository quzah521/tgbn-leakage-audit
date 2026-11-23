# Label-window probe: tgbn-trade

```
tgbn-trade: unit u = year, weighting = abs
  single-offset cosine peaks at a = 0 (cos 1.000)
  label window (window search (highest cosine among windows with full row coverage)): [L + 0u, L + 1u)  cos 1.000, row coverage 1.000
  -> label(L) is available from L + 1u
  documented window: [L + 0u, L + 1u)
  - label window matches the documented window
  - label(L) is complete only at L + 1 units; using it as an input before then is look-ahead
```

## Single offsets

Cosine between the label at L and the node's edge vector in [L + a·u, L + (a+1)·u); row coverage = share of labelled nodes with at least one edge there.

| offset a | row coverage | cos_count | cos_abs |
|---|---|---|---|
| -8 | 0.808 | 0.230 | 0.518 |
| -7 | 0.857 | 0.245 | 0.561 |
| -6 | 0.908 | 0.263 | 0.610 |
| -5 | 0.911 | 0.265 | 0.628 |
| -4 | 0.968 | 0.287 | 0.690 |
| -3 | 0.968 | 0.292 | 0.716 |
| -2 | 0.975 | 0.301 | 0.759 |
| -1 | 0.983 | 0.312 | 0.809 |
| 0 (peak) | 1.000 | 0.375 | 1.000 |
| 1 | 0.984 | 0.307 | 0.817 |
| 2 | 0.979 | 0.296 | 0.767 |
| 3 | 0.976 | 0.286 | 0.736 |
| 4 | 0.924 | 0.264 | 0.673 |
| 5 | 0.872 | 0.244 | 0.618 |
| 6 | 0.818 | 0.226 | 0.567 |
| 7 | 0.762 | 0.206 | 0.517 |
| 8 | 0.710 | 0.190 | 0.468 |

## Best contiguous windows around the peak

| window [L+a0·u, L+(a1+1)·u) | row coverage | cos_abs |
|---|---|---|
| [L+0, L+1) | 1.000 | 1.000 |
| [L+0, L+2) | 1.000 | 0.953 |
| [L+-1, L+1) | 1.000 | 0.951 |
| [L+-1, L+2) | 1.000 | 0.940 |
| [L+-1, L+3) | 1.000 | 0.924 |
