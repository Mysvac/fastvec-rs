# Benchmark Results for Container Operations

## Quick Start

Clone the repository, navigate to this folder, and run:

```bash
cargo bench
```

## Existing Reports

For current benchmark results, refer to [reports](./reports/report/index.html).

- `Vec` – The standard library `Vec` container.
- `ArrayVec` – The `ArrayVec` container provided by this crate.

- `FastVecData` – Represents typical `FastVec` usage: obtaining the internal `FastVecData` handle and performing operations through it. In this mode, the cached pointer works correctly, eliminating branch overhead.

- `FastVec` – Represents usage where the `data()` method is called to obtain the `FastVecData` reference for each operation. This incurs pointer refresh overhead on every access.

- `fastvec::SmallVec` – The standard SBO (Small Buffer Optimization) dynamic array from this crate.
- `smallvec::SmallVec` – The implementation from the `smallvec` crate.

## Key Findings

For small datasets that require reallocation, `Vec` is significantly slower than containers with inline buffers. The following comparisons focus on performance **excluding reallocation overhead**.

### Random Access – Small Datasets

`ArrayVec` == `Vec` == `FastVecData` < `fastvec::SmallVec` < `FastVec` ≈≈ `smallvec::SmallVec`

- `FastVecData` incurs no overhead due to its cached pointer.
- `fastvec::SmallVec` uses `#[cold]` to mark cold paths, improving branch prediction for data stored inline.

### Random Access – Large Datasets

`Vec` == `FastVecData` < `FastVec` ≈≈ `smallvec::SmallVec` < `fastvec::SmallVec`

> `ArrayVec` is typically unsuitable for large datasets (fully inline, risks stack overflow).

- `fastvec::SmallVec` suffers from poor branch prediction for heap-stored data due to its `#[cold]` path marking.

### Insertion & Deletion – Small Datasets (Excluding Reallocation)

`ArrayVec` == `Vec` < `fastvec::SmallVec` <= `FastVecData` <= `FastVec` <= `smallvec::SmallVec`

- `FastVecData` still incurs branch checks without cold-path optimization, making it slightly less efficient than `fastvec::SmallVec`, though the difference is marginal.

### Insertion & Deletion – Large Datasets (Excluding Reallocation)

> `ArrayVec` is typically unsuitable for large datasets (fully inline, risks stack overflow).

`Vec` < `FastVecData` < `smallvec::SmallVec` ≈≈ `FastVec` < `fastvec::SmallVec`

---

## Summary Table

| Operation / Dataset | Fastest → Slowest |
|---------------------|------------------|
| Random Access (Small) | `ArrayVec` / `Vec` / `FastVecData` → `fastvec::SmallVec` → `FastVec` / `smallvec::SmallVec` |
| Random Access (Large) | `Vec` / `FastVecData` → `FastVec` / `smallvec::SmallVec` → `fastvec::SmallVec` |
| Insert/Delete (Small) | `ArrayVec` / `Vec` → `fastvec::SmallVec` → `FastVecData` → `FastVec` → `smallvec::SmallVec` |
| Insert/Delete (Large) | `Vec` → `FastVecData` → `smallvec::SmallVec` / `FastVec` → `fastvec::SmallVec` |