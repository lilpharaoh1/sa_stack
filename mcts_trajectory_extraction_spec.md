# MCTS Trajectory Extraction Specification

## Overview

After running MCTS for `n` iterations, we need to extract `k` diverse candidate trajectories from the resulting tree. The current implementation only returns a single best trajectory. We want to extract multiple trajectories that represent diverse, promising action sequences explored during search.

## Required Behaviour

### 1. MCTS runs as normal for `n` iterations

Standard MCTS loop: selection (UCB), expansion, evaluation (rollout or value estimate), backup (update Q and N). No changes here.

### 2. After search completes, extract `k` trajectories via DFS

Once the `n` iterations are done and the tree is built:

- Perform a **depth-first search (DFS)** starting from the root node `s0`.
- At each internal node, visit children in **decreasing order of visit count `N(s, a)`** — i.e., most-visited (most promising) children first.
- Collect the first `k` **leaf nodes** encountered by this DFS. Call them `{l1, l2, ..., lk}`.
- A leaf node is any node that has no expanded children, OR a node at maximum tree depth.

### 3. Pad each leaf to full trajectory length

Each leaf `l_i` only covers the portion of the trajectory within the tree. To get a full-horizon trajectory:

- From each leaf `l_i`, run a **padding policy** `π_padding` forward until the terminal condition is met (e.g., planning horizon `H` is reached).
- The padding policy can be a simple heuristic (e.g., IDM, constant speed, or any default policy). It does NOT need to be the same as the rollout policy used during MCTS search.

### 4. Assemble full trajectories

For each leaf `l_i`:

- The trajectory `τ_i` is the **full sequence of (state, action) pairs from root `s0` down through the tree to `l_i`**, concatenated with the **(state, action) pairs from the padding rollout**.
- This gives `k` complete trajectories from `s0` to the terminal state.

### 5. Return all `k` trajectories

Return the set `{τ_1, τ_2, ..., τ_k}` as candidate trajectories. A downstream scorer/selector will choose among them.

## Key Implementation Details

- The DFS ordering by decreasing `N(s, a)` is critical. It ensures the most-explored (and therefore most promising) branches are visited first, while still collecting diverse trajectories from different parts of the tree.
- The `k` leaves will naturally span different branches of the tree, giving diversity. Early leaves will be from the most-visited (best) branches; later leaves will be from less-visited but still explored branches.
- If the tree has fewer than `k` leaves, return as many as exist.
- The padding policy does NOT use the MCTS tree — it is a simple forward simulation from the leaf state.

## What This Achieves

- **Focused diversity**: all `k` trajectories come from regions of the trajectory space that MCTS has evaluated as somewhat promising (they were explored during search), but they represent different action sequences (different branches of the tree).
- **Not just the single best**: instead of `argmax_a Q(s0, a)` returning one trajectory, we get `k` trajectories spanning the promising region of the action space.

## Pseudocode

```
function EXTRACT_TRAJECTORIES(root, k, π_padding):
    trajectories = []
    stack = [(root, [])]  # (node, path_so_far)
    
    while stack is not empty AND len(trajectories) < k:
        node, path = stack.pop()
        
        if node is leaf (no expanded children) OR node is at max depth:
            # Pad from this leaf to full horizon
            padded = RUN_PADDING_POLICY(node.state, π_padding)
            full_trajectory = path + padded
            trajectories.append(full_trajectory)
        else:
            # Get children sorted by visit count N, ASCENDING
            # (because stack is LIFO, ascending push = descending pop)
            children = sorted(node.children, key=lambda c: N(node.state, c.action))
            for child in children:  # ascending N, so highest N popped first
                stack.append((child, path + [(node.state, child.action)]))
    
    return trajectories
```

**Important**: the DFS stack ordering matters. Since a stack is LIFO, push children in ascending order of `N` so that the child with the highest `N` is on top and gets popped (visited) first.
