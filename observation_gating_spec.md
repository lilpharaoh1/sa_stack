# Observation Gating for Kalman Update

## Problem

When the human's action doesn't distinguish between aware and unaware configurations (e.g., the other vehicle is far away and doesn't affect the optimal policy), the log likelihood ratio y ≈ 0. The Kalman update incorrectly treats this as evidence that ψ ≈ 0, pulling the estimate toward zero and shrinking P. We want to skip the update when the observation is uninformative.

## Implementation

In the outer-loop Kalman observation update, after computing the log likelihood ratio y for each participant, gate the update based on informativeness:

```python
def kalman_update_gated(psi_hat, P_diag, y, R, min_informativeness=0.1):
    n = len(psi_hat)
    psi_hat_new = psi_hat.copy()
    P_diag_new = P_diag.copy()
    
    for i in range(n):
        informativeness = abs(y[i])
        
        if informativeness > min_informativeness:
            # Observation is informative, run normal Kalman update
            K = P_diag[i] / (P_diag[i] + R)
            psi_hat_new[i] = psi_hat[i] + K * (y[i] - psi_hat[i])
            P_diag_new[i] = (1 - K) * P_diag[i]
        # else: skip update, keep psi_hat and P unchanged
    
    return psi_hat_new, P_diag_new
```

## Informativeness Definition

`informativeness = |y_i| = |log(L_aware / L_unaware)|`

This is the absolute log likelihood ratio. It measures how much more likely the observed human action is under one configuration versus the other:

- `|y| ≈ 0`: the action is equally likely under both configurations → uninformative
- `|y| > 0`: the action favours one configuration over the other → informative

## Parameter

`min_informativeness`: threshold below which the update is skipped. Start with 0.1. If the estimate still drifts toward zero when it shouldn't, increase it. If the estimate is too slow to respond to genuinely informative actions, decrease it.

## Where to Apply

This replaces the existing `kalman_update` call in the outer planning loop (step 8 in the full planning loop from the previous spec). Everything else stays the same.
