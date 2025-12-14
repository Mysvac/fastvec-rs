## Intro

A high-performance vector library that stores small collections on the stack
and automatically spills to the heap when needed.

Similar to `SmallVec`, but we split the responsibility into two distinct containers instead of one.

Many workloads have small collections that fit comfortably in a fixed-size stack buffer,
but occasionally need to grow larger. Stack allocation is much faster than heap allocation,
especially for cache locality and allocator overhead.

FastVec solves this by keeping data on the stack initially and transparently moving to the heap
only when necessary—with zero cost for the common case.

## Containers

### `StackVec`

- **Fixed capacity** on the stack
- **Array-like** performance
- **Vec-like** interface
- **Panics** if capacity is exceeded
- Use when: You know the maximum size in advance

```rust
# use fastvec::StackVec;
let mut vec: StackVec<i32, 10> = StackVec::new();
assert_eq!(vec.capacity(), 10);

vec.push(1);
vec.push(2);
assert_eq!(vec.len(), 2);
// Cannot push more than 10 items (will panic)
```

### `AutoVec`

- **Flexible capacity**: stack initially, heap when needed
- **Enum-based**: internally either `StackVec` or `Vec`
- **Never panics** from capacity limits
- Use when: Size is unknown but usually small.

```rust
# use fastvec::{AutoVec, autovec};
let mut vec: AutoVec<i32, 5> = autovec![1, 2, 3];
assert!(vec.in_stack());  // Still on stack

// Push beyond capacity—automatically migrates to heap
vec.extend(&[4, 5, 6, 7, 8]);
assert!(!vec.in_stack()); // Now on heap
```

Note: When the amount of data is large, this is not as good as [`Vec`]
because its operation usually requires an additional judgment.

## Comparison

| Feature | StackVec | AutoVec | SmallVec | Vec |
|---------|----------|---------|----------|-----|
| Stack storage | ✓ | ✓ | ✓ | ✗ |
| Flexible capacity | ✗ | ✓ | ✓ | ✓ |
| Speed | A | B | B | B~C |

See detailed documentation in [`StackVec`] and [`AutoVec`] for method signatures and examples.

### Alias

- `MiniVec<T>` = `AutoVec<T, 8>` — for tiny collections
- `FastVec<T>` = `AutoVec<T, 16>` — general-purpose balance

## `no_std` support

This crate requires only `core` and `alloc`, making it suitable for embedded and no_std environment

## Optional features

### `serde`

When this optional dependency is enabled,
`StackVec` and `AutoVec` implements the `serde::Serialize` and `serde::Deserialize` traits.

