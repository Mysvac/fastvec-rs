# Benchmark Results for Container Operations

## Overview

This benchmark suite compares the performance of four Rust vector containers across common operations. Results are collected using criterion.rs and represent real-world scenarios with randomized data to minimize compiler optimizations.

See report in [here](http://static.mysvac.com/fastvec/report).

`FastVec` is for typical usage scenarios, where data operations are performed through a `FastVecData`.
`FastVec_Direct` is for cases where each operation re-acquires a mutable reference to the handle -
for example, every push is done via `fastvec.get_mut().push(..)`.

If you are unable to access the link, please refer to the text summary below.

## Quick Start

Clone the entire project, enter this folder, and then run:

```bash
cargo bench
```

## Tested Containers

| Container | Description |
|-----------|-------------|
| `Vec` | Standard heap-allocated vector |
| `StackVec` | Stack-only vector with fixed capacity |
| `FastVec` | Optimized vector for data processing |
| `AutoVec` | Small-size optimized vector |
| `SmallVec` | Small-size optimized vector |

## Tested Operations

- **`new/with_capacity`** - Initialization
- **`push`** - Element insertion at end
- **`pop`** - Element removal from end
- **`insert`** - Element insertion at index
- **`remove`** - Element removal at index
- **`Index`** - Random access

## Methodology

To ensure realistic benchmarks, we use randomized data and avoid excessive compiler optimizations.
This helps simulate real-world workloads rather than best-case scenarios.

## Results & Analysis

### Initialization (`new`/`with_capacity`)

- All containers show similar performance when heap allocation is required.
- When there is no need to allocate heap memory, `StackVec`, `FastVec`, `AutoVec`, and `SmallVec` are equally fast, can be completed during compile-time.

---

### Random Access (`Index`)

Negligible performance difference

**Takeaway**: Index performance is dominated by CPU cache behavior, not container implementation.

---

### Push & Pop (End Operations)

**Performance Ranking:**
1. [`StackVec`] - fastest
2. [`Vec`] & [`FastVec`] (without reallocation) - very close
3. [`SmallVec`] & [`AutoVec`] - slower

**Why [`SmallVec`] & [`AutoVec`] lags**: It must check whether data resides on the heap or stack during every operation. This conditional is cheap individually, but cumulative overhead becomes noticeable.

**Why [`FastVec`] competes with [`Vec`]**: It avoids [`SmallVec`]'s conditional by caching pointers, trading a small amount of code complexity for performance.

**Takeaway**: Simple operations are dominated by branch prediction costs. Stack-only containers win, and smart caching beats conditional checking.

---

### Insert & Remove (Middle Operations)

**Winner: Virtually tied (including [`StackVec`])** ✓

- **[`StackVec`]**: Marginally faster, but difference is negligible
- **Others**: Nearly identical performance

**Why no clear winner**: The dominant cost is data movement (shifting elements), not container management. The conditional cost that hurts [`SmallVec`] in push/pop operations becomes invisible here.

**Takeaway**: When algorithmic complexity dominates, container overhead becomes irrelevant.

---

### Memory Growth Strategy

[`StackVec`] is stack only.

[`FastVec`], [`AutoVec`], [`SmallVec`], and [`Vec`] are all consistent, with double growth.

