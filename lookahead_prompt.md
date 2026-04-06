# Lookahead Intervention Controllers

I need two new intervention controllers added to my existing ACC simulation. The environment, human model, particle filter, and reactive controller already work.

## Controller interface

Same as the existing reactive controller:
- Input: current state s_k, human's intended action a_H, current belief over kappa (particles + weights)
- Output: executed action a_k

## Human model (already exists)

```
v_desired = kappa * v_lead + k_d * (d - d_star)
a_H = k_v * (v_desired - v_ego)
```

## Controller 1: Lookahead (certainty-equivalent)

Over a horizon of N steps, find the action sequence that minimises total squared deviation from the human's predicted actions while maintaining d >= d_safe throughout.

Key: the human's predicted action at each future step depends on the state at that step, which depends on the executed actions up to that point. So you must interleave: execute action → update state → predict human action at new state → repeat.

```
kappa_hat = weighted mean of particles

for j in range(N):
    a_H_pred[j] = human_policy(s_pred[j], kappa_hat)
    a_exec[j] = decision variable
    s_pred[j+1] = dynamics(s_pred[j], a_exec[j])

minimise: sum_j (a_exec[j] - a_H_pred[j])^2
subject to: d_pred[j] >= d_safe for all j
            v_pred[j] >= 0 for all j
```

Apply only a_exec[0], discard the rest (receding horizon).

Use scipy.optimize.minimize with SLSQP, or a simple iterative shooting method — simulate forward, clip any actions that violate constraints, repeat until convergence.

## Controller 2: Lookahead (robust)

Same as Controller 1, but the safety constraint must hold under the worst-case kappa from the belief. Take the top 3 particles by weight. For each, simulate the trajectory forward and require d >= d_safe under all of them. The objective still uses kappa_hat for predicting the human's actions.

## Parameters

- N = 20 (horizon steps)
- d_safe = 5m
- dt = 0.1s
- All other parameters (k_d, k_v, d_star) are already defined in the environment.
