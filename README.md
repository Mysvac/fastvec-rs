# FastVec: Fast, Stack-Optimized Vectors

A high-performance vector library optimized for small collections. Data is stored on the stack
initially and automatically migrates to the heap only when capacity is exceeded.

> See more infomation in [`Document`](https://docs.rs/fastvec/latest/fastvec/) and [`crate.io`](https://crates.io/crates/fastvec) .

## Why Stack Allocation?

Many real-world workloads operate on small collections that fit comfortably in a fixed-size
stack buffer, but occasionally need to grow larger. Stack allocation is significantly faster
than heap allocation due to:
- **Zero allocator overhead**: No malloc/free calls for small data
- **Better cache locality**: Stack data stays in L1/L2 cache
- **Predictable performance**: No allocation latency spikes

## The FastVec Approach

Unlike [`SmallVec`](https://docs.rs/smallvec/latest/smallvec), which combines both behaviors
in a single container, FastVec uses a **two-container strategy**:
- **`StackVec`**: For fixed-size stacks (zero overhead, maximum performance)
- **`FastVec`**: For flexible growth (stack-to-heap migration with pointer caching)

This design achieves **higher efficiency** by eliminating runtime checks in the hot path.
See [benchmark results](https://github.com/Mysvac/fastvec-rs/blob/main/benches/README.md) for detailed comparisons.

## Container Guide

| Container | Storage | Growth | Use Case |
|-----------|---------|--------|----------|
| **`StackVec`** | Stack only (fixed) | Fixed capacity | Size is known and bounded |
| **`FastVec`** | Stack → Heap (dynamic) | Automatic | Size uncertain but usually small |

### `StackVec`: The High-Performance Container

A vector with fixed capacity backed entirely by the stack.

**Features:**
- ✓ Fixed capacity (compile-time configurable)
- ✓ Zero heap allocations
- ✓ Array-like performance
- ✓ Vec-like interface
- ✗ Panics if capacity is exceeded

**Best for:** Known, bounded collection sizes (e.g., small buffers, fixed arrays).

```rust
let mut vec: StackVec<i32, 10> = StackVec::new();
vec.push(1);
vec.push(2);
assert_eq!(vec.len(), 2);
assert_eq!(vec.capacity(), 10); // Fixed capacity
```

### `FastVec`: The Flexible Container

A vector that starts with stack storage and transparently migrates to the heap when needed.

**Features:**
- ✓ Stack storage for small collections (default: 8 elements)
- ✓ Automatic heap migration when needed
- ✓ Never panics from capacity limits
- ✓ Vec-like interface
- ✓ Pointer caching for efficiency (eliminates runtime checks)

**Best for:** Uncertain collection sizes that are usually small (e.g., parsed data, result collections).

```rust
// Default stack capacity is 8
let mut vec: FastVec<_> = fastvec![1, 2, 3];
assert!(vec.in_cache()); // Still on stack

// Customize stack capacity
let mut vec: FastVec<i32, 5> = fastvec![1, 2, 3];

// Grow beyond stack capacity → automatically migrates to heap
vec.get_mut().extend([4, 5, 6, 7, 8]);
assert!(!vec.in_cache()); // Now on heap
```

Unlike `SmallVec`, `FastVec` caches a pointer to the current data,
eliminating the conditional check ("is this on stack or heap?") from the critical path.

This results in measurably faster operations on small collections, but requires explicit
access through `get_ref` or `get_mut` for certain
operations. See the `FastVec` documentation for details on the API trade-offs and usage patterns.

## `no_std` Support

FastVec requires only `core` and `alloc`, making it ideal for embedded systems and no_std environments.
Full freestanding Rust support is included by default.

## Optional Features

### `serde`

When enabled, both `StackVec` and `FastVec` implement:
- [`serde::Serialize`](https://docs.rs/serde/latest/serde/trait.Serialize.html)
- [`serde::Deserialize`](https://docs.rs/serde/latest/serde/trait.Deserialize.html)

### `nightly`

Only available in Nightly version.

This will enable the `cold_path` feature, further
optimizing the `push`, `pop`, `insert` performance of `FastVec`.
