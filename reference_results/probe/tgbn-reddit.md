# Label-window probe: tgbn-reddit

```
tgbn-reddit: unit u = day, weighting = count
  single-offset cosine peaks at a = 5 (cos 0.939)
  label window (window search (highest cosine among windows with full row coverage)): [L + 5u, L + 6u)  cos 0.939, row coverage 1.000
  -> label(L) is available from L + 6u
  documented window: [L + 0u, L + 7u)
  - label window covers only part of the documented window (0, 6)
  - label(L) is complete only at L + 6 units; using it as an input before then is look-ahead
```

## Single offsets

Cosine between the label at L and the node's edge vector in [L + a·u, L + (a+1)·u); row coverage = share of labelled nodes with at least one edge there.

| offset a | row coverage | cos_count | cos_abs |
|---|---|---|---|
| -8 | 0.665 | 0.189 | 0.171 |
| -7 | 0.649 | 0.181 | 0.163 |
| -6 | 0.667 | 0.192 | 0.173 |
| -5 | 0.698 | 0.195 | 0.177 |
| -4 | 0.677 | 0.196 | 0.177 |
| -3 | 0.673 | 0.198 | 0.177 |
| -2 | 0.704 | 0.206 | 0.184 |
| -1 | 0.699 | 0.208 | 0.189 |
| 0 | 0.678 | 0.202 | 0.179 |
| 1 | 0.687 | 0.202 | 0.184 |
| 2 | 0.724 | 0.215 | 0.194 |
| 3 | 0.718 | 0.223 | 0.202 |
| 4 | 0.719 | 0.258 | 0.238 |
| 5 (peak) | 1.000 | 0.939 | 0.846 |
| 6 | 0.747 | 0.266 | 0.232 |
| 7 | 0.706 | 0.216 | 0.195 |
| 8 | 0.730 | 0.217 | 0.192 |

## Best contiguous windows around the peak

| window [L+a0·u, L+(a1+1)·u) | row coverage | cos_count |
|---|---|---|
| [L+5, L+6) | 1.000 | 0.939 |
| [L+4, L+6) | 1.000 | 0.810 |
| [L+5, L+7) | 1.000 | 0.795 |
| [L+3, L+6) | 1.000 | 0.736 |
| [L+4, L+7) | 1.000 | 0.735 |
