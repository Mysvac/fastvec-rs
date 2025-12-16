# Benchmark Results for Container Operations

## Overview

This benchmark suite compares the performance of four Rust vector containers across common operations. Results are collected using criterion.rs and represent real-world scenarios with randomized data to minimize compiler optimizations.

See report in [here](http://static.mysvac.com/fastvec/report).

`FastVec` is for typical usage scenarios, where data operations are performed through a `FastVecData`.
`FastVec_Direct` is for cases where each operation re-acquires a mutable reference to the handle -
for example, every push is done via `fastvec.get_mut().push(..)`.

In some scenarios, the compiler may optimize the repeated borrows,
making the performance close to that of a single borrowed reference.

If you are unable to access the link, please refer to the text summary below.

## Quick Start

Clone this project and run:

```bash
cargo bench
```

## Tested Containers

| Container | Description |
|-----------|-------------|
| [`Vec`] | Standard heap-allocated vector |
| [`FastVec`] | Optimized vector with intelligent memory caching |
| [`StackVec`] | Stack-only vector with fixed capacity |
| [`SmallVec`] | Small-size optimized vector |

## Tested Operations

- **[`new/with_capacity`](Vec::with_capacity)** - Initialization
- **[`push`](Vec::push)** - Element insertion at end
- **[`pop`](Vec::pop)** - Element removal from end
- **[`insert`](Vec::insert)** - Element insertion at index
- **[`remove`](Vec::remove)** - Element removal at index
- **[`Index`](core::ops::Index)** - Random access

## Methodology

To ensure realistic benchmarks, we use randomized data and avoid excessive compiler optimizations. This helps simulate real-world workloads rather than best-case scenarios.

## Results & Analysis

### Initialization (`new`/`with_capacity`)

**Winner: [`StackVec`]** ⭐

- **[`StackVec`]**: Fixed O(1) compile-time initialization (no allocations)
- **[`FastVec`] & [`SmallVec`]**: Use stack storage for small capacities, also completing at compile-time
- **[`Vec`] & heap allocations**: All containers show similar performance when heap allocation is required

**Takeaway**: Stack-based containers excel at initialization, especially for predictable, small capacities.

> The `new_large` result of this test is very inaccurate, and the application time for larger memory is very random.
> Even if the sampling rate and time are extended, the result will be different each time.

---

### Random Access (`Index`)

**Winner: Virtually tied** (marginal difference)

- **[`StackVec`]**: Microscopically faster due to simpler implementation
- **Others**: Negligible performance difference

**Takeaway**: Index performance is dominated by CPU cache behavior, not container implementation.

---

### Push & Pop (End Operations)

**Winner: [`StackVec`]** ⭐ **Runner-up: [`Vec`] & [`FastVec`]**

**Performance Ranking:**
1. [`StackVec`] - fastest
2. [`Vec`] & [`FastVec`] (without reallocation) - very close
3. [`SmallVec`] - slower

**Why [`SmallVec`] lags**: It must check whether data resides on the heap or stack during every operation. This conditional is cheap individually, but cumulative overhead becomes noticeable.

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

[`FastVec`] is consistent with [`Vec`] and [`SmallVec`], double the growth.

---

## Summary

| Use Case | Best Choice | Why |
|----------|-------------|-----|
| **Fixed-size, small collections** | [`StackVec`] | Zero allocation, predictable performance |
| **General-purpose, standard workloads** | [`Vec`] or [`FastVec`] | Proven, simple (Vec) or memory-frugal (FastVec) |
| **Memory-constrained with small bias** | [`SmallVec`] | Good for small data, but conditional overhead hurts simple ops |
| **Memory frugal with flexibility** | [`FastVec`] | Good for small data, the operation is slightly complicated. |

