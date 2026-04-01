//! A high-performance vector crate tuned for small data sizes.
//!
//! Uses small-buffer optimization (SBO): data is stored in an inline buffer first,
//! then moved to heap storage when capacity is exceeded.
//!
//! ## Container Guide
//!
//! We provide three containers for different scenarios:
//!
//! | Container | Storage | Best for |
//! |-----------|---------|----------|
//! | **[ArrayVec]** | Inline-only, fixed capacity | When you need peak performance and know the max element count |
//! | **[FastVec]** | Inline first, auto-switch to heap | When you need peak performance for temporary data |
//! | **[SmallVec]** | Inline first, auto-switch to heap | When you need long-term storage with an unknown but typically small element count |
//!
//! If you have many elements and need long-term storage, consider using [`Vec`](alloc::vec::Vec) directly.
//!
//! ### [ArrayVec]
//!
//! An inline-only `Vec` that allocates space without initializing data.
//!
//! **Features:**
//! - No extra heap allocations
//! - Extreme array-like performance
//! - Vec-compatible API
//! - Compile-time fixed capacity, cannot grow
//!
//! A great replacement for a plain array `[T; N]`.
//!
//! ```rust, ignore
//! # use fastvec::ArrayVec;
//! let mut vec: ArrayVec<i32, 10> = ArrayVec::new();
//!
//! vec.push(1);
//! vec.push(2);
//!
//! assert_eq!(vec, [1, 2]);
//! assert_eq!(vec.len(), 2);
//! assert_eq!(vec.capacity(), 10); // Fixed capacity
//! ```
//!
//! Supports nearly all [`Vec`](alloc::vec::Vec) operations (except reallocation).
//!
//! See the [`ArrayVec`] docs for details.
//!
//! ### [FastVec]
//!
//! A `Vec` for temporary data that auto-grows. It prefers inline storage and switches
//! to the heap when capacity is insufficient.
//!
//! **Features:**
//! - `!Sync`, generally for temporary data processing
//! - Supports capacity growth
//! - Inline-first; no heap allocs for small sizes
//! - Guaranteed not slower than `Vec` for large sizes
//!
//! This container caches pointers to minimize inline/heap checks, keeping performance
//! from degrading (even on the heap it’s no slower than `Vec`) and outperforming [`SmallVec`].
//!
//! ```rust, ignore
//! # use fastvec::FastVec;
//! let mut vec: FastVec<i32, 5> = [1, 2, 3].into();
//! assert_eq!(vec.capacity(), 5);
//!
//! // Auto-grows; switches to heap when needed
//! vec.data().extend([4, 5, 6, 7, 8]);
//! assert_eq!(vec, [1, 2, 3, 4, 5, 6, 7, 8])
//! ```
//!
//! Pointer caching introduces self-references, so it is `!Sync`. Any data access must
//! first call [`FastVec::data`] to obtain the correct [`FastVecData`] reference.
//!
//! `data` incurs one branch and pointer assignment; typically you should grab the data
//! reference once, use it via references, and only switch when you need to move the data.
//!
//! See the [`FastVec`] docs for details.
//!
//! ### [SmallVec]
//!
//! A space-optimized SBO `Vec`.
//!
//! **Features:**
//! - `Sync + Send`, suitable for long-term storage
//! - Supports capacity growth
//! - Inline-first; no heap allocs for small sizes
//! - Vec-compatible API
//!
//! Unlike [`FastVec`], this container checks inline/heap location on operations
//! and is designed similarly to [`smallvec::SmallVec`](https://docs.rs/smallvec/latest/smallvec).
//!
//! It is efficient for small data but may lag `Vec` on large data,
//! especially on simple functions like data access and `push/pop`.
//!
//! ```rust, ignore
//! # use fastvec::SmallVec;
//! let mut vec: SmallVec<i32, 5> = [1, 2, 3].into();
//! assert_eq!(vec.capacity(), 5);
//!
//! // Auto-grows; switches to heap when needed
//! vec.extend([4, 5, 6, 7, 8]);
//! assert_eq!(vec, [1, 2, 3, 4, 5, 6, 7, 8])
//! ```
//!
//! Supports all [`Vec`](alloc::vec::Vec) operations without needing `data()`.
//!
//! See the [`SmallVec`] docs for details.
//!
//! ## no_std Support
//!
//! FastVec depends only on `core` and `alloc` by default,
//! making it ideal for embedded and no_std environments.
//!
//! ## Optional Features
//!
//! ### arrayvec
//!
//! Enabled by default. If disabled, `ArrayVec` code is not compiled.
//!
//! ### fastvec
//!
//! Enabled by default. If disabled, `FastVec` code is not compiled.
//!
//! ### smallvec
//!
//! Enabled by default. If disabled, `SmallVec` code is not compiled.
//!
//! ### serde
//!
//! When enabled, `ArrayVec`, `FastVec` and `SmallVec` implement
//! `serde::Serialize` and `serde::Deserialize` .
//!
//! ### std
//!
//! When enabled, `ArrayVec`, `FastVec` and `SmallVec` implement `std::io::Write` .
#![cfg_attr(docsrs, feature(doc_cfg))]
#![no_std]

extern crate alloc;

#[cfg(any(feature = "fastvec", feature = "arrayvec", feature = "smallvec",))]
mod utils;

#[cfg(feature = "fastvec")]
pub mod fast;

#[cfg(feature = "fastvec")]
pub use fast::{FastVec, FastVecData};

#[cfg(feature = "arrayvec")]
pub mod array;

#[cfg(feature = "arrayvec")]
pub use array::ArrayVec;

#[cfg(feature = "smallvec")]
pub mod small;

#[cfg(feature = "smallvec")]
pub use small::SmallVec;

#[cfg(feature = "serde")]
mod serde;

#[cfg(feature = "std")]
mod std_io;
