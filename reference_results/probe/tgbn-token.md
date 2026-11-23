# Label-window probe: tgbn-token

```
tgbn-token: unit u = day, weighting = abs
  single-offset cosine peaks at a = 5 (cos 0.965)
  label window (window search (highest cosine among windows with full row coverage)): [L + 5u, L + 6u)  cos 0.965, row coverage 1.000
  -> label(L) is available from L + 6u
  documented window: [L + 0u, L + 7u)
  - label window covers only part of the documented window (0, 6)
  - label(L) is complete only at L + 6 units; using it as an input before then is look-ahead
```

## Single offsets

Cosine between the label at L and the node's edge vector in [L + a·u, L + (a+1)·u); row coverage = share of labelled nodes with at least one edge there.

| offset a | row coverage | cos_count | cos_abs |
|---|---|---|---|
| -8 | 0.408 | 0.272 | 0.272 |
| -7 | 0.383 | 0.255 | 0.255 |
| -6 | 0.377 | 0.260 | 0.261 |
| -5 | 0.385 | 0.272 | 0.272 |
| -4 | 0.380 | 0.251 | 0.252 |
| -3 | 0.404 | 0.276 | 0.276 |
| -2 | 0.461 | 0.327 | 0.327 |
| -1 | 0.474 | 0.335 | 0.336 |
| 0 | 0.453 | 0.313 | 0.313 |
| 1 | 0.457 | 0.329 | 0.331 |
| 2 | 0.451 | 0.327 | 0.327 |
| 3 | 0.464 | 0.344 | 0.345 |
| 4 | 0.513 | 0.384 | 0.385 |
| 5 (peak) | 1.000 | 0.949 | 0.965 |
| 6 | 0.552 | 0.421 | 0.424 |
| 7 | 0.477 | 0.345 | 0.347 |
| 8 | 0.490 | 0.346 | 0.347 |

## Best contiguous windows around the peak

| window [L+a0·u, L+(a1+1)·u) | row coverage | cos_abs |
|---|---|---|
| [L+5, L+6) | 1.000 | 0.965 |
| [L+5, L+7) | 1.000 | 0.933 |
| [L+4, L+6) | 1.000 | 0.928 |
| [L+5, L+8) | 1.000 | 0.909 |
| [L+3, L+6) | 1.000 | 0.908 |
