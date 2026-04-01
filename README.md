# FastVec: SBO Vector

A dynamic array crate optimized for small data.

Small-buffer optimization (SBO) stores a tiny inline buffer on the stack and
avoids heap allocation when the element count stays small. This improves
runtime in the common case while still allowing growth.

Benchmark results are available in [benchmark/README.md](benchmark/README.md).

---

## ArrayVec

**Fully inlined dynamic array** with a compile-time fixed maximum capacity.

```rust
let mut vec: ArrayVec<i32, 10> = ArrayVec::new();

vec.push(1);
vec.push(2);

assert_eq!(vec, [1, 2]);
assert_eq!(vec.len(), 2);
assert_eq!(vec.capacity(), 10); // Fixed capacity
```

Similar to `[T; N]`, but uses a `len` field to track valid elements and allows
uninitialized trailing slots. The API largely mirrors `Vec`. Any operation that
exceeds capacity will panic.

You can also store it as `Box<ArrayVec<_, N>>` to keep the fixed capacity on the heap.

### Features

- Zero overhead: array-like access with no branches
- No heap allocations: data is fully inlined
- Fixed capacity: cannot grow beyond `N`

This container is similar to `arrayvec`'s `ArrayVec`.

### Feature flag

Disable the crate `arrayvec` feature (enabled by default) to avoid compiling it.

---

## SmallVec

**Classic SBO vector** with a branch to decide whether data is inline or on the heap.

```rust
let mut vec: SmallVec<i32, 2> = SmallVec::new();

vec.push(1);  // Inline storage
vec.push(2);  // Inline storage
vec.push(3);  // Exceeds capacity, moves to heap

assert_eq!(vec, [1, 2, 3]);
```

### Features

- Space efficient: `SmallVec<u64, 2>` has the same size as `Vec<u64>`
- Automatic growth: switches to heap without user involvement
- Cold-path optimization: heap path is marked `#[cold]`
- Branch on access: each access checks storage location
- Heap access is slightly slower for large data

This container is similar to `smallvec`'s `SmallVec`. Compared to it, this
crate marks the heap path as `#[cold]`, improving small-size access speed while
being slightly slower for large data. See [benchmark/README.md](benchmark/README.md)
for details.

### Feature flag

Disable the crate `smallvec` feature (enabled by default) to avoid compiling it.

---

## FastVec

**Speed-optimized SBO vector** using a cached pointer to avoid per-access branching.

```rust
let mut vec: FastVec<i32, 2> = FastVec::new();
let data = vec.data();

data.push(1);
data.push(2);
data.push(3);  // Moves to heap automatically

assert_eq!(data.as_slice(), &[1, 2, 3]);
```

`FastVec` keeps a cached pointer to the current storage, so data access is
as fast as `Vec` regardless of inline or heap storage. Growing and shrinking
still requires checks (lightweight; see benchmarks).

### Features

- Constant access speed: matches `Vec` on both stack and heap
- Heap-friendly: no performance drop after growing
- Zero extra branch on access
- Self-referential: internal pointer may point to stack frame; must be refreshed
- Not `Sync`: uses `Cell` internally

### How it works

`FastVec` uses a two-type design to handle self-references:

- `FastVec`: movable wrapper that refreshes the pointer
- `FastVecData`: operational handle obtained via `FastVec::data()`

```rust
let mut vec: FastVec<i32, 4> = [1, 2, 3].into();
let data = vec.data();

data.push(4);
data.retain(|x| *x % 2 == 0);
```

### Feature flag

Disable the crate `fastvec` feature (enabled by default) to avoid compiling it.

---

## Selection Guide

| Feature | ArrayVec | SmallVec | FastVec |
|---------|----------|----------|---------|
| **Capacity** | Fixed, no growth | Grows to heap | Grows to heap |
| **Space efficiency** | High | High | Medium (extra pointer) |
| **Access overhead** | Zero | Branch per access | Zero (cached pointer) |
| **Heap performance** | N/A | Lower (cold path) | Same as `Vec` |
| **Sync** | Yes | Yes | No |

- Max size known, peak performance: use `ArrayVec`
- Usually small, space-sensitive storage: use `SmallVec`
- Usually small, performance-sensitive temporary data: use `FastVec`

## no_std Support

FastVec depends only on `core` and `alloc`, making it ideal for embedded and no_std environments.

## Optional Features

### arrayvec

Enabled by default. If disabled, `ArrayVec` code is not compiled.

### fastvec

Enabled by default. If disabled, `FastVec` code is not compiled.

### smallvec

Enabled by default. If disabled, `SmallVec` code is not compiled.

### serde

When enabled, `ArrayVec`, `FastVec` and `SmallVec` implement
`serde::Serialize` and `serde::Deserialize`.

### std

When enabled, `ArrayVec`, `FastVec` and `SmallVec` implement `std::io::Write`.
