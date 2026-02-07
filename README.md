# FastVec: Stack-Optimized High-Speed Vector

A high-performance vector crate tuned for small data sizes.

Favors stack-backed storage to reduce heap allocations and improve CPU cache locality.

## Container Guide

We provide three containers for different scenarios:

| Container | Storage | Best for |
|-----------|---------|----------|
| **StackVec** | Stack-only, fixed capacity | When you need peak performance and know the max element count |
| **FastVec** | Stack first, auto-switch to heap | When you need peak performance for temporary data |
| **AutoVec** | Stack first, auto-switch to heap | When you need long-term storage with an unknown but typically small element count |

If you have many elements and need long-term storage, consider using `Vec` directly.

### StackVec

A stack-resident `Vec` that allocates space without initializing data.

**Features:**
- No heap allocations
- Extreme array-like performance
- Vec-compatible API
- Compile-time fixed capacity, cannot grow

A great replacement for a plain array `[T; N]`.

```rust
let mut vec: StackVec<i32, 10> = StackVec::new();

vec.push(1);
vec.push(2);

assert_eq!(vec, [1, 2]);
assert_eq!(vec.len(), 2);
assert_eq!(vec.capacity(), 10); // Fixed capacity
```

Supports nearly all `Vec` operations (except reallocation).

See the `StackVec` docs for details.

### FastVec

A `Vec` for temporary data that auto-grows. It prefers the stack and switches to the heap when capacity is insufficient.

**Features:**
- `!Sync`, generally for temporary data processing
- Supports capacity growth
- Stack-first; no heap allocs for small sizes
- Guaranteed not slower than `Vec` for large sizes

This container caches pointers to minimize stack/heap checks, keeping performance from degrading (even on the heap it’s no slower than `Vec`) and outperforming [`SmallVec`](https://docs.rs/smallvec/latest/smallvec).

```rust
let mut vec: FastVec<i32, 5> = fastvec![1, 2, 3];
assert_eq!(vec.capacity(), 5);

// Auto-grows; switches to heap when needed
vec.data().extend([4, 5, 6, 7, 8]);
assert!(!vec.in_stack()); // Now on heap
assert_eq!(vec, [1, 2, 3, 4, 5, 6, 7, 8])
```

Pointer caching introduces self-references, so it is `!Sync`. Any data access must first call `data` to obtain the correct `FastVecData` reference.

`data` incurs one branch and pointer assignment; typically you should grab the data reference once, use it via references, and only switch when you need to move the data.

See the `FastVec` docs for details.

### AutoVec

A small-data-optimized `Vec` for long-term storage, implemented as an enum of `Vec` and `StackVec`.

**Features:**
- `Sync + Send`, suitable for long-term storage
- Supports capacity growth
- Stack-first; no heap allocs for small sizes
- Vec-compatible API

Unlike `FastVec`, this container checks stack/heap location on operations
and is designed similarly to [`SmallVec`](https://docs.rs/smallvec/latest/smallvec).
It is efficient for small data but may lag `Vec` on large data,
especially on simple functions like `push/pop`.

```rust
let mut vec: AutoVec<i32, 5> = autovec![1, 2, 3];
assert_eq!(vec.capacity(), 5);

// Auto-grows; switches to heap when needed
vec.extend([4, 5, 6, 7, 8]);
assert!(!vec.in_stack()); // Now on heap
assert_eq!(vec, [1, 2, 3, 4, 5, 6, 7, 8])
```

Supports all `Vec` operations without needing `data()`.

See the `AutoVec` docs for details.

## no_std Support

FastVec depends only on `core` and `alloc`, making it ideal for embedded and no_std environments.

## Optional Features

### serde

When enabled, `StackVec`, `FastVec` and `AutoVec` implement
`serde::Serialize` and `serde::Deserialize` .

### std

When enabled, `StackVec`, `FastVec` and `AutoVec` implement `std::io::Write` .

### nightly

Available only on Nightly.

When enabled, the `cold_path` feature is used to optimize branch prediction.
