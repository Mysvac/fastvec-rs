//! ## Intro
//!
//! A high-performance vector library that stores small collections on the stack
//! and automatically spills to the heap when needed.
//!
//! Similar to [`SmallVec`], but we split the responsibility into two distinct containers instead of one.
//!
//! Many workloads have small collections that fit comfortably in a fixed-size stack buffer,
//! but occasionally need to grow larger. Stack allocation is much faster than heap allocation,
//! especially for cache locality and allocator overhead.
//!
//! FastVec solves this by keeping data on the stack initially and transparently moving to the heap
//! only when necessary—with zero cost for the common case.
//!
//! ## Containers
//!
//! ### `StackVec`
//!
//! - **Fixed capacity** on the stack
//! - **Array-like** performance
//! - **Vec-like** interface
//! - **Panics** if capacity is exceeded
//! - Use when: You know the maximum size in advance
//!
//! ```
//! # use fastvec::StackVec;
//! let mut vec: StackVec<i32, 10> = StackVec::new();
//! assert_eq!(vec.capacity(), 10);
//!
//! vec.push(1);
//! vec.push(2);
//! assert_eq!(vec.len(), 2);
//! // Cannot push more than 10 items (will panic)
//! ```
//!
//! ### `AutoVec`
//!
//! - **Flexible capacity**: stack initially, heap when needed
//! - **Enum-based**: internally either `StackVec` or `Vec`
//! - **Vec-like** interface
//! - **Never panics** from capacity limits
//! - Use when: Size is unknown but usually small.
//!
//! ```
//! # use fastvec::{AutoVec, autovec};
//! let mut vec: AutoVec<i32, 5> = autovec![1, 2, 3];
//! assert!(vec.in_stack());  // Still on stack
//!
//! // Push beyond capacity—automatically migrates to heap
//! vec.extend(&[4, 5, 6, 7, 8]);
//! assert!(!vec.in_stack()); // Now on heap
//! ```
//!
//! Note: When the amount of data is large, this is not as good as [`Vec`]
//! because its operation usually requires an additional judgment.
//!
//! ## Comparison
//!
//! | Feature | StackVec | AutoVec | SmallVec | Vec |
//! |---------|----------|---------|----------|-----|
//! | Stack storage | ✓ | ✓ | ✓ | ✗ |
//! | Flexible capacity | ✗ | ✓ | ✓ | ✓ |
//! | Speed | A | B | B | B~C |
//!
//! See detailed documentation in [`StackVec`] and [`AutoVec`] for method signatures and examples.
//!
//! ### Alias
//!
//! - [`MiniVec<T>`] = `AutoVec<T, 8>` — for tiny collections
//! - [`FastVec<T>`] = `AutoVec<T, 16>` — general-purpose balance
//!
//! ## `no_std` support
//!
//! This crate requires only `core` and `alloc`, making it suitable for embedded and no_std environments.
//!
//! ## Optional features
//!
//! ### `serde`
//!
//! When this optional dependency is enabled,
//! [`StackVec`] and [`AutoVec`] implements the [`serde::Serialize`] and [`serde::Deserialize`] traits.
//!
//!
//! [`serde::Serialize`]: https://docs.rs/smallvec/latest/serde
//! [`serde::Deserialize`]: https://docs.rs/smallvec/latest/serde
//! [`SmallVec`]: https://docs.rs/smallvec/latest/smallvec
//! [`Vec`]: alloc::vec::Vec
#![no_std]

extern crate alloc;

mod utils;

pub mod stack_vec;

#[cfg(feature = "serde")]
mod serde;

#[doc(inline)]
pub use stack_vec::StackVec;

pub mod auto_vec;
#[doc(inline)]
pub use auto_vec::AutoVec;

/// A small `AutoVec` with a stack capacity of 8 elements.
///
/// This is an alias for [`AutoVec<T, 8>`].
///
/// `MiniVec` is optimized for scenarios where you expect small collections most of the time,
/// typically 8 or fewer elements. It provides zero-cost stack allocation for small data
/// and automatically spills to the heap when capacity is exceeded.
///
/// # Examples
///
/// ```
/// # use fastvec::MiniVec;
/// let mut vec: MiniVec<i32> = MiniVec::new();
///
/// // Small collections stay on the stack with no heap allocation
/// vec.push(1);
/// vec.push(2);
/// vec.push(3);
/// assert!(vec.in_stack());
/// assert_eq!(vec, [1, 2, 3]);
///
/// // Exceeding capacity automatically moves to heap
/// vec.extend(&[4, 5, 6, 7, 8, 9]);
/// assert!(!vec.in_stack());
/// assert_eq!(vec.len(), 9);
/// ```
pub type MiniVec<T> = AutoVec<T, 8>;

/// A fast `AutoVec` with a stack capacity of 16 elements.
///
/// This is an alias for [`AutoVec<T, 16>`].
///
/// `FastVec` is a balanced choice between stack efficiency and flexibility,
/// suitable for most general-purpose use cases. It can hold 16 elements on the stack
/// before spilling to the heap, making it ideal for collections that frequently stay small
/// but occasionally grow larger.
///
/// # Examples
///
/// ```
/// # use fastvec::FastVec;
/// let mut vec: FastVec<String> = FastVec::new();
///
/// // Moderate collections typically remain on the stack
/// vec.push("hello".to_string());
/// vec.push("world".to_string());
/// assert!(vec.in_stack());
///
/// // Can be extended up to 16 elements without heap allocation
/// for i in 0..14 {
///     vec.push(format!("item_{}", i));
/// }
/// assert!(vec.in_stack());
/// assert_eq!(vec.len(), 16);
///
/// // Beyond 16 elements, automatically uses heap
/// vec.push("beyond".to_string());
/// assert!(!vec.in_stack());
/// ```
pub type FastVec<T> = AutoVec<T, 16>;
