use alloc::{boxed::Box, vec::Vec};
use core::fmt;
use core::iter::FusedIterator;
use core::mem::ManuallyDrop;
use core::pin::Pin;
use core::{marker::PhantomData, ptr};

use crate::StackVec;
pub use crate::fast_vec_data::FastVecData;

/// A stack-prioritized vector that automatically spills to the heap when capacity is exceeded.
///
/// [`FastVec`] is optimized for small collections: it stores data on the stack initially and
/// transparently migrates to the heap when needed. This eliminates heap allocation overhead
/// for the common case of small data.
///
/// Unlike [`SmallVec`](https://docs.rs/smallvec/latest/smallvec/), [`FastVec`] uses **pointer caching**
/// to avoid conditional checks on every operation, achieving higher performance. However, this design
/// requires accessing data through [`FastVecData`].
///
/// # Quick Start
///
/// ## Creating a FastVec
///
/// Creating a [`FastVec`] is similar to `SmallVec`:
///
/// ```
/// # use fastvec::FastVec;
/// // Default stack capacity is 8 elements
/// let mut vec = FastVec::<i32>::new();
/// assert_eq!(vec.capacity(), 8);
/// assert_eq!(vec.len(), 0);
///
/// // Customize stack capacity via const generic
/// let mut vec = FastVec::<i32, 12>::new();
/// assert!(vec.in_cache()); // Data is on stack
///
/// // If requested capacity exceeds N, data goes to heap
/// let mut vec = FastVec::<i32, 2>::with_capacity(4);
/// assert!(!vec.in_cache()); // Data is on heap
/// ```
///
/// ## Modifying Data
///
/// Most data-modifying operations require obtaining a [`FastVecData`] via
/// [`get_mut`](FastVec::get_mut) (for mutable access) or [`get_ref`](FastVec::get_ref) (for read-only access):
///
/// ```
/// # use fastvec::{FastVec, fast_vec::FastVecData};
/// let mut vec: FastVec<_> = [1, 2, 3, 4].into();
/// let data: &mut FastVecData<_,_> = vec.get_mut();
///
/// // Use it like a Vec
/// data.push(5);
/// data.insert(0, 6);
/// # let v = data.as_slice();
/// assert_eq!(v, &[6, 1, 2, 3, 4, 5]);
/// ```
///
/// # API Design
///
/// [`FastVec`] supports nearly all [`Vec`] methods, categorized as follows:
///
/// ## Operations Through [`FastVecData`]
///
/// Operations that take `&self` or `&mut self` require:
/// - [`push`](FastVecData::push), [`pop`](FastVecData::pop)
/// - [`insert`](FastVecData::insert), [`remove`](FastVecData::remove)
/// - [`drain`](FastVecData::drain), [`extract_if`](FastVecData::extract_if)
/// - And more...
///
/// **Why?** See the `Understanding FastVecData` section below.
///
/// ## Operations Directly on [`FastVec`]
///
/// Consuming or conversion operations can be called directly:
/// - [`into_vec`](FastVec::into_vec), [`into_boxed_slice`](FastVec::into_boxed_slice)
/// - [`IntoIterator`], [`From`] conversions
/// - And more...
///
/// ```
/// # use fastvec::FastVec;
/// let vec: FastVec<_> = [1, 2, 3, 4].into();
/// let boxed: Box<[i32]> = vec.into_boxed_slice();
/// ```
///
/// ## Convenience Methods on [`FastVec`]
///
/// A few frequently-used APIs are exposed directly on [`FastVec`] for convenience:
/// - [`len`](FastVec::len), [`capacity`](FastVec::capacity), [`is_empty`](FastVec::is_empty)
/// - [`as_slice`](FastVec::as_slice), [`as_mut_slice`](FastVec::as_mut_slice)
///
/// [`as_slice`](FastVec::as_slice) and [`as_mut_slice`](FastVec::as_mut_slice)
/// internally call [`get_ref`](FastVec::get_ref) or [`get_mut`](FastVec::get_mut) first.
///
/// ## Trait Implementations
///
/// [`FastVec`] implements [`Deref`](core::ops::Deref), [`Index`](core::ops::Index),
/// [`Debug`](core::fmt::Debug), etc., via [`as_slice`](FastVec::as_slice) and [`as_mut_slice`](FastVec::as_mut_slice):
///
/// ```
/// # use fastvec::FastVec;
/// let mut vec: FastVec<_> = [1, 4, 3, 2].into();
/// vec.sort(); // via Deref
///
/// assert_eq!(vec[1], 2); // via Index
/// assert_eq!(vec, [1, 2, 3, 4]); // via PartialEq
/// ```
///
/// **Performance note:** These operations refresh the data ptr each time. For complex operations
/// like `sort`, this overhead is negligible. However, for simple operations (`Index`, `push`, `pop`),
/// the overhead is measurable.
///
/// ## Recommended Usage Pattern
///
/// For best performance, acquire the data reference once and reuse it:
///
/// ```
/// # use fastvec::FastVec;
/// let mut vec: FastVec<_> = [1, 4, 3, 2].into();
/// let data = vec.get_mut();
///
/// // All operations reuse the same reference
/// data.sort();
/// data.push(5);
/// assert_eq!(data, &[1, 2, 3, 4, 5]);
///
/// // Use FastVec only when you need to create/move/consume it
/// let vec: Vec<_> = vec.into_vec();
/// assert_eq!(vec, [1, 2, 3, 4, 5]);
/// ```
///
/// The real advantage of [`FastVec`] lies in data processing rather than storage,
/// and it is usually recommended to convert it to [`Vec`] when transferring data.
///
/// If [`FastVec`]'s data is already in the heap, this conversion only requires copying pointers, which is very cheap.
/// If it is on the stack, it is equivalent to only applying for heap memory once, won't be more expensive than using [`Vec`].
///
/// # Understanding `FastVecData`
///
/// ## The Problem
///
/// A naive stack-to-heap vector looks like this:
///
/// ```ignore
/// struct NaiveVec<T, const N: usize> {
///     cache: [MaybeUninit<T>; N],
///     heap_ptr: *mut T,
///     len: usize,
///     cap: usize,
///     in_cache: bool, // Is data on stack or heap?
/// }
/// ```
///
/// **Problem:** Every operation (`push`, `pop`, `index`, etc.) must check `in_cache` to determine
/// whether to access `cache` or `heap_ptr`. This conditional is cheap individually but cumulative
/// overhead becomes significant for simple operations.
///
/// ## The Solution: Pointer Caching
///
/// Make a single pointer always point to the current data location:
///
/// ```ignore
/// struct FastVecData<T, const N: usize> {
///     cache: [MaybeUninit<T>; N],
///     ptr: Cell<*mut T>,  // Always points to active data
///     len: usize,
///     cap: usize,
///     in_cache: bool,     // Only checked during reallocation
/// }
/// ```
///
/// Now `ptr` directly accesses data without branching. The `in_cache` check is only needed when
/// resizing capacity, not on every operation.
///
/// But when data is on the stack, `ptr` points to `cache`—creating a **self-referential structure**.
/// Moving [`FastVecData`] invalidates `ptr`, which must be "refreshed" (repointed to `cache`).
///
/// ## The Design: Two-Type Architecture
///
/// [`FastVec`] is a thin wrapper around [`FastVecData`]:
/// - **[`FastVec`]**: Manages the pointer refresh logic; can be freely moved
/// - **[`FastVecData`]**: Performs actual data operations; accessed only through borrows
///
/// When you call [`get_ref`](FastVec::get_ref) or [`get_mut`](FastVec::get_mut), [`FastVec`]:
/// 1. Refreshes the pointer (if data is on stack)
/// 2. Returns a borrow of [`FastVecData`]
///
/// Rust's borrow checker ensures [`FastVecData`] cannot be moved while borrowed, so the pointer
/// remains valid during handle usage.
///
/// ## Why [`Cell`]?
///
/// Pointer refresh needs interior mutability (even [`get_ref`](FastVec::get_ref) must update the pointer).
/// We use [`Cell`](core::cell::Cell) instead of atomic operations because:
/// - Atomic pointers add runtime overhead on every read
/// - Cross-platform atomic pointer support varies
/// - Single-threaded refresh is sufficient (handles are `Sync`)
///
/// # Zero-Sized Types (ZST)
///
/// [`FastVec`] fully supports ZSTs. For zero-sized types:
/// - `ptr` is a dangling pointer; refresh operations are no-ops (optimized away by compiler)
/// - No stack or heap memory is allocated, regardless of element count
/// - The generic parameter `N` remains semantically meaningful: [`in_cache`](FastVec::in_cache)
///   and [`capacity`](FastVec::capacity) behave as if space were allocated
///
/// This ensures API consistency across all types while maintaining zero overhead for ZSTs.
///
/// # Thread Safety
///
/// **[`FastVec`]**: Implements [`Send`] but **not** [`Sync`] due to internal [`Cell`](core::cell::Cell) usage
/// (required for pointer relocation). Concurrent calls to [`get_ref`](FastVec::get_ref) may race.
///
/// - **[`FastVecData`]**: Implements both [`Send`] and [`Sync`], so you can safely share its reference across threads.
///
/// It's not recommended to store this struct in [`Arc`](alloc::sync::Arc),
/// and it is better to use [`Vec`] directly when heap memory is already in use.
///
/// But if you really want to do this, see [`FastVec::into_pinned_box`].
#[repr(transparent)]
pub struct FastVec<T, const N: usize = 8> {
    inner: FastVecData<T, N>,
    _marker: PhantomData<*const ()>,
}

// All functions have a dependency on [`FastVecData::refresh`], but it doesn't seem thread safe.
// unsafe impl<T, const N: usize> Sync for FastVecData<T, N> where T: Sync {}
unsafe impl<T, const N: usize> Send for FastVec<T, N> where T: Send {}

/// Creates a [`FastVec`] containing the arguments.
///
/// The syntax is similar to [`vec!`](https://doc.rust-lang.org/std/macro.vec.html).
///
/// # Examples
///
/// ```
/// # use fastvec::{fastvec, FastVec};
/// let vec: FastVec<String, 10> = fastvec![];
/// let vec: FastVec<i64, 10> = fastvec![1; 5]; // Need to support Clone.
/// let vec: FastVec<_, 10> = fastvec![1, 2, 3, 4];
/// ```
#[macro_export]
macro_rules! fastvec {
    [] => { $crate::FastVec::new() };
    [$elem:expr; $n:expr] => { $crate::FastVec::from_elem($elem, $n) };
    [$($item:expr),+ $(,)?] => { $crate::FastVec::from_buf([ $($item),+ ]) };
}

impl<T, const N: usize> FastVec<T, N> {
    /// Constructs a new, empty [`FastVec`] on the stack with the specified capacity.
    ///
    /// The capacity must be provided at compile time via the const generic parameter, default is `8`.
    ///
    /// Note that the capacity should not be too large to avoid stack overflow.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let vec: FastVec<i32, 8> = FastVec::new();
    ///
    /// // equivalent to this
    /// let vec: FastVec<i32> = [].into();
    /// ```
    #[inline]
    pub const fn new() -> Self {
        Self {
            inner: unsafe { FastVecData::new() },
            _marker: PhantomData,
        }
    }

    /// Constructs a new, empty [`FastVec`] with at least the specified capacity.
    ///
    /// If the specified capacity is less than or equal to `N`, this is equivalent to [`new`](FastVec::new),
    /// and no heap memory will be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    ///
    /// let vec: FastVec<i32, 5> = FastVec::with_capacity(4);
    /// assert!(vec.in_cache());
    ///
    /// let vec: FastVec<i32, 5> = FastVec::with_capacity(10);
    /// assert!(!vec.in_cache());
    /// ```
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: unsafe { FastVecData::with_capacity(capacity) },
            _marker: PhantomData,
        }
    }

    /// Creates a [`FastVec`] directly from a pointer, a length, and a capacity.
    ///
    /// This does not copy data; it sets pointers and lengths directly and treats the data as heap-allocated.
    ///
    /// See more information in [`Vec::from_raw_parts`].
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *mut T, length: usize, capacity: usize) -> Self {
        Self {
            inner: unsafe { FastVecData::from_raw_parts(ptr, length, capacity) },
            _marker: PhantomData,
        }
    }

    /// Returns `true` if the vector contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<i32> = FastVec::new();
    /// assert!(vec.is_empty());
    ///
    /// vec.get_mut().push(1);
    /// assert!(!vec.is_empty());
    /// ```
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.inner.len == 0
    }

    /// Returns the number of elements in the vector, also referred to as its length.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<i32, 8> = [1, 2, 3, 4].into();
    /// assert_eq!(vec.capacity(), 8);
    /// assert_eq!(vec.len(), 4);
    ///
    /// vec.get_mut().extend([1, 2, 3,  4, 5]);
    /// assert!(vec.capacity() >= 9);
    /// assert_eq!(vec.len(), 9);
    /// ```
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.inner.len
    }

    /// Returns the total number of elements the vector can hold without reallocating.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<i32, 8> = [1, 2, 3, 4].into();
    /// assert_eq!(vec.capacity(), 8);
    /// assert_eq!(vec.len(), 4);
    ///
    /// vec.get_mut().extend([1, 2, 3,  4, 5]);
    /// assert!(vec.capacity() >= 9);
    /// assert_eq!(vec.len(), 9);
    /// ```
    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        self.inner.cap
    }

    /// Returns `true` if the data is stored in the stack cache.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<i32, 8> = [1, 2, 3, 4].into();
    /// assert!(vec.in_cache());
    ///
    /// vec.get_mut().extend([1, 2, 3,  4, 5]);
    /// assert!(!vec.in_cache());
    /// ```
    #[inline(always)]
    pub const fn in_cache(&self) -> bool {
        self.inner.in_cache
    }

    /// Check and refresh the pointer to ensure it points to the correct location.
    ///
    /// Note that although this crate reduces calls in many places,
    /// **the overhead is very low**, with only one branch and one pointer assignment.
    ///
    /// This function usually does not need to be called manually;
    /// other methods call it when needed.
    ///
    /// This is internal mutability, and `Sync` is disabled because it may not be thread safe.
    #[inline(always)]
    pub fn refresh(&self) {
        unsafe {
            self.inner.refresh();
        }
    }

    /// Refresh the pointer and return a reference to the internal data.
    ///
    /// Although [`FastVec`] itself is not [`Sync`],
    /// [`FastVecData`] is, so you can use this method to obtain a reference
    /// to [`FastVecData`] and pass it between threads or closures.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<i32, 8> = [1, 2, 3, 4].into();
    /// let mut x = 2;
    ///
    /// // let mut p = || { x += vec[0]; }; // Err
    ///
    /// let v = vec.get_ref();
    /// let mut p = || { x += v[0]; }; // Ok
    ///
    /// p();
    /// assert_eq!(x, 3);
    /// ```
    #[inline]
    pub fn get_ref(&self) -> &FastVecData<T, N> {
        self.refresh();
        &self.inner
    }

    /// Refresh the pointer and return a mutable reference to the internal data.
    ///
    /// You can use this mutable reference for methods such as `push`, `pop`, `retain`, and `insert`.
    /// The pointer is refreshed once in `get_mut`; later method calls reuse it without extra cost.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<i32, 8> = [1, 2, 3, 4].into();
    /// let v = vec.get_mut();
    ///
    /// v.push(5);
    /// v.retain(|x| *x % 2 == 1);
    ///
    /// assert_eq!(vec, [1, 3, 5]);
    /// ```
    #[inline]
    pub fn get_mut(&mut self) -> &mut FastVecData<T, N> {
        self.refresh();
        &mut self.inner
    }

    /// Refresh the pointer and obtain slices of the data.
    ///
    /// During the validity period of the slice reference, the data will not be moved, so the pointer is valid.
    ///
    /// This enables [`FastVec`] to implement many traits directly through slice access.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<i32, 8> = [1, 4, 3, 2].into();
    ///
    /// vec.sort(); // `Deref` trait, internal impl with `as_mut_slice`.
    ///
    /// let x = vec[1]; /// `Index` trait, internal impl with `as_slice`.
    ///
    /// assert_eq!(x, 2);
    /// ```
    ///
    /// Method cost depends on implementation: for `sort` the refresh cost is negligible,
    /// while `Index`-style operations may effectively double the work.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.refresh();
        self.inner.as_slice()
    }

    /// Refresh the pointer and obtain mutable slices of the data.
    ///
    /// During the slice's lifetime, the data will not move, so the pointer remains valid.
    ///
    /// This enables [`FastVec`] to implement many traits directly through slice access.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<i32, 8> = [1, 4, 3, 2].into();
    ///
    /// vec.sort(); // `Deref` trait, internal impl with `as_mut_slice`.
    ///
    /// let x = vec[1]; /// `Index` trait, internal impl with `as_slice`.
    ///
    /// assert_eq!(x, 2);
    /// ```
    ///
    /// Method cost depends on implementation: for `sort` the refresh cost is negligible,
    /// while `Index`-style operations may effectively double the work.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.refresh();
        self.inner.as_mut_slice()
    }

    /// Convert [`FastVec`] to [`Vec`].
    ///
    /// If the data is in the stack, the exact memory will be allocated.
    /// If the data is already on the heap, no reallocation is needed.
    ///
    /// The returned [`Vec`] may not be tight because heap data does not shrink here.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let vec: FastVec<i32, 3> = [1, 2].into();
    /// assert!(vec.in_cache());
    /// let vec: Vec<_> = vec.into_vec();
    /// assert_eq!(vec, [1, 2]);
    /// assert!(vec.capacity() == 2);
    ///
    /// let vec: FastVec<i32, 3> = [1, 2, 3, 4, 5].into();
    /// assert!(!vec.in_cache());
    /// let vec: Vec<_> = vec.into_vec();
    /// assert_eq!(vec, [1, 2, 3, 4, 5]);
    /// assert!(vec.capacity() >= 5);
    /// ```
    pub fn into_vec(self) -> Vec<T> {
        self.refresh();
        self.inner.into_vec()
    }

    /// Convert [`FastVec`] to [`Vec`].
    ///
    /// If the data is in the stack, the exact memory will be allocated.
    /// If the data is in the heap, [`Vec::shrink_to_fit`] will be called.
    pub fn shrink_into_vec(self) -> Vec<T> {
        self.refresh();
        self.inner.shrink_into_vec()
    }

    /// Convert [`FastVec`] to [`Box<[T]>`].
    pub fn into_boxed_slice(self) -> Box<[T]> {
        self.refresh();
        self.inner.into_boxed_slice()
    }

    /// Convert [`FastVec`] to a leaked slice.
    ///
    /// See [`Vec::leak`].
    pub fn leak<'a>(self) -> &'a mut [T] {
        self.refresh();
        self.inner.leak()
    }

    /// Convert [`FastVec<T, N>`] to [`FastVec<T, P>`].
    ///
    /// - If the data is in the heap area, it will not be reallocated.
    /// - If the data is in the stack area, it needs to be copied.
    ///
    /// No data is lost.
    pub fn force_cast<const P: usize>(mut self) -> FastVec<T, P> {
        if self.inner.in_cache {
            unsafe {
                let state = <FastVec<T, P>>::from_raw_parts(
                    self.get_mut().as_mut_ptr(),
                    self.len(),
                    self.capacity(),
                );
                self.inner.len = 0;
                state
            }
        } else {
            let mut state = <FastVec<T, P>>::with_capacity(self.inner.len);
            let dst = state.get_mut();
            let src = self.get_mut();
            unsafe {
                ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), src.len);
                dst.len = src.len;
                src.len = 0;
            }
            state
        }
    }

    /// Create a vector with a specified number of elements,
    /// cloning from the provided value.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let vec = FastVec::<i32, 5>::from_elem(1, 3);
    ///
    /// assert_eq!(vec, [1, 1, 1]);
    /// ```
    pub fn from_elem(value: T, num: usize) -> Self
    where
        T: Clone,
    {
        let mut state = Self::with_capacity(num);
        if num > 0 {
            let vec = state.get_mut();
            unsafe {
                for _ in 1..num {
                    vec.push_unchecked(value.clone());
                }
                vec.push_unchecked(value);
            }
        }
        state
    }

    /// Initialize values from a fixed-length array.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let vec: FastVec<_> = FastVec::from([1, 2, 3]);
    ///
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    #[inline(always)]
    pub fn from_buf<const P: usize>(values: [T; P]) -> Self {
        Self::from(values)
    }

    /// Move all data to the heap and return a pinned handle.
    ///
    /// [`FastVec`] is **not [`Sync`]** (uses [`Cell`](core::cell::Cell)), so shared
    /// references cannot be sent across threads. If you need cross-thread sharing,
    /// either use a static value or move the data to the heap first.
    ///
    /// In most cases, prefer converting to [`Vec`] via [`FastVec::into_vec`]
    /// for the simplest, most efficient heap representation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// # use std::sync::{Arc, RwLock};
    /// let vec: FastVec<i32, 8> = [1, 4, 3, 2].into();
    /// let mut pin_box = vec.into_pinned_box();
    ///
    /// pin_box.sort();
    /// pin_box.push(5);
    /// assert_eq!(*pin_box, [1, 2, 3, 4, 5]);
    ///
    /// let arc = Arc::new(RwLock::new(pin_box));
    /// // Use `arc` across threads...
    /// ```
    #[inline]
    pub fn into_pinned_box(self) -> Pin<Box<FastVecData<T, N>>> {
        let vec = Box::pin(self.inner);
        unsafe {
            vec.refresh();
        }
        vec
    }
}

impl<T, const N: usize, const P: usize> FastVec<[T; P], N> {
    /// Takes a `FastVec<[T; P], N>` and flattens it into a `FastVec<T, S>`.
    ///
    /// If the data is stored on the heap, this only requires converting the pointer type (no reallocation).
    /// If it's on the stack, data is copied.
    ///
    /// If stack capacity is insufficient, [`Vec`] will be used.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<_, 2> = [[1, 2, 3], [4, 5, 6], [7, 8, 9]].into();
    /// assert_eq!(vec.get_mut().pop(), Some([7, 8, 9]));
    /// assert!(!vec.in_cache());
    ///
    /// let mut flattened = vec.into_flattened::<6>();
    /// assert_eq!(flattened, [1, 2, 3, 4, 5, 6]);
    /// assert!(!flattened.in_cache());
    ///
    /// let mut vec: FastVec<_, 3> = [[1, 2, 3], [4, 5, 6], [7, 8, 9]].into();
    /// assert_eq!(vec.get_mut().pop(), Some([7, 8, 9]));
    /// assert!(vec.in_cache());
    ///
    /// let mut flattened = vec.into_flattened::<5>();
    /// assert_eq!(flattened, [1, 2, 3, 4, 5, 6]);
    /// assert!(!flattened.in_cache());
    /// ```
    pub fn into_flattened<const S: usize>(self) -> FastVec<T, S> {
        FastVec {
            inner: self.inner.into_flattened(),
            _marker: PhantomData,
        }
    }
}

impl<T, const N: usize> Default for FastVec<T, N> {
    /// Constructs a new, empty `FastVec` on the stack with the specified capacity.
    ///
    /// Equivalent to [`FastVec::new`].
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone, const N: usize> Clone for FastVec<T, N> {
    fn clone(&self) -> Self {
        let src = self.get_ref();
        let mut vec = Self::with_capacity(src.len);
        let dst = vec.get_mut();
        for item in src.as_slice() {
            unsafe {
                dst.push_unchecked(item.clone());
            }
        }
        vec
    }

    fn clone_from(&mut self, source: &Self) {
        let dst = self.get_mut();
        let src = source.get_ref();
        dst.clear();
        dst.reserve(src.len);

        for item in src.as_slice() {
            unsafe {
                dst.push_unchecked(item.clone());
            }
        }
    }
}

impl<T, const N: usize, const P: usize> From<StackVec<T, P>> for FastVec<T, N> {
    fn from(mut value: StackVec<T, P>) -> Self {
        let len = value.len();
        let mut vec = Self::with_capacity(len);
        let vec_mut = vec.get_mut();
        unsafe {
            ptr::copy_nonoverlapping(value.as_ptr(), vec_mut.as_mut_ptr(), len);
            vec_mut.len = len;
            value.set_len(0);
        }
        vec
    }
}

impl<T, const N: usize> From<Vec<T>> for FastVec<T, N> {
    fn from(mut value: Vec<T>) -> Self {
        let len = value.len();
        let mut vec = Self::with_capacity(len);
        let vec_mut = vec.get_mut();
        unsafe {
            ptr::copy_nonoverlapping(value.as_ptr(), vec_mut.as_mut_ptr(), len);
            vec_mut.len = len;
            value.set_len(0);
        }
        vec
    }
}

impl<T, const N: usize> From<Box<[T]>> for FastVec<T, N> {
    fn from(value: Box<[T]>) -> Self {
        let mut res = Self::with_capacity(value.len());
        let vec = res.get_mut();
        for items in value {
            unsafe {
                vec.push_unchecked(items);
            }
        }
        res
    }
}

impl<T, const N: usize, const P: usize> From<[T; P]> for FastVec<T, N> {
    fn from(value: [T; P]) -> Self {
        let mut vec = Self::with_capacity(P);
        let vec_mut = vec.get_mut();
        unsafe {
            ptr::copy_nonoverlapping(value.as_ptr(), vec_mut.as_mut_ptr(), P);
            vec_mut.len = P;
            drop(value);
        }
        vec
    }
}

impl<T: Clone, const N: usize> From<&[T]> for FastVec<T, N> {
    fn from(value: &[T]) -> Self {
        let mut res = Self::with_capacity(value.len());
        let vec = res.get_mut();
        for items in value {
            unsafe {
                vec.push_unchecked(items.clone());
            }
        }
        res
    }
}

impl<T: Clone, const N: usize, const P: usize> From<&[T; P]> for FastVec<T, N> {
    fn from(value: &[T; P]) -> Self {
        let mut res = Self::with_capacity(value.len());
        let vec = res.get_mut();
        for items in value {
            unsafe {
                vec.push_unchecked(items.clone());
            }
        }
        res
    }
}

impl<T: Clone, const N: usize> From<&mut [T]> for FastVec<T, N> {
    #[inline]
    fn from(value: &mut [T]) -> Self {
        <Self as From<&[T]>>::from(value)
    }
}

impl<T: Clone, const N: usize, const P: usize> From<&mut [T; P]> for FastVec<T, N> {
    #[inline]
    fn from(value: &mut [T; P]) -> Self {
        <Self as From<&[T; P]>>::from(value)
    }
}

impl<T, const N: usize> FromIterator<T> for FastVec<T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (hint, _) = iter.size_hint();
        let mut res = Self::with_capacity(hint);
        let vec = res.get_mut();
        for item in iter {
            vec.push(item);
        }
        res
    }
}

crate::utils::impl_commen_traits!(FastVec<T, N>);

impl<T, U, const N: usize> PartialEq<FastVec<U, N>> for FastVec<T, N>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &FastVec<U, N>) -> bool {
        PartialEq::eq(self.as_slice(), other.as_slice())
    }
}

impl<'a, T: Clone, const N: usize> From<&'a FastVec<T, N>> for alloc::borrow::Cow<'a, [T]> {
    fn from(v: &'a FastVec<T, N>) -> alloc::borrow::Cow<'a, [T]> {
        alloc::borrow::Cow::Borrowed(v.as_slice())
    }
}

impl<'a, T: Clone, const N: usize> From<FastVec<T, N>> for alloc::borrow::Cow<'a, [T]> {
    fn from(v: FastVec<T, N>) -> alloc::borrow::Cow<'a, [T]> {
        alloc::borrow::Cow::Owned(v.into_vec())
    }
}

/// An iterator that consumes a [`FastVec`] and yields its items by value.
#[derive(Clone)]
pub struct IntoIter<T, const N: usize> {
    vec: ManuallyDrop<FastVec<T, N>>,
    index: usize,
}

unsafe impl<T, const N: usize> Send for IntoIter<T, N> where T: Send {}
unsafe impl<T, const N: usize> Sync for IntoIter<T, N> where T: Sync {}

impl<T, const N: usize> IntoIterator for FastVec<T, N> {
    type Item = T;
    type IntoIter = IntoIter<T, N>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            vec: ManuallyDrop::new(self),
            index: 0,
        }
    }
}

impl<T, const N: usize> Iterator for IntoIter<T, N> {
    type Item = T;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.vec.inner.len {
            self.index += 1;
            self.vec.refresh();
            unsafe { Some(ptr::read(self.vec.inner.as_ptr().add(self.index - 1))) }
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let v = self.vec.inner.len - self.index;
        (v, Some(v))
    }
}

impl<T, const N: usize> DoubleEndedIterator for IntoIter<T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let len = self.vec.inner.len;
        if self.index < len {
            self.vec.inner.len -= 1;
            self.vec.refresh();
            unsafe { Some(ptr::read(self.vec.inner.as_ptr().add(len - 1))) }
        } else {
            None
        }
    }
}

impl<T, const N: usize> ExactSizeIterator for IntoIter<T, N> {
    #[inline]
    fn len(&self) -> usize {
        self.vec.inner.len - self.index
    }
}

impl<T, const N: usize> FusedIterator for IntoIter<T, N> {}

impl<T: fmt::Debug, const N: usize> fmt::Debug for IntoIter<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("IntoIter")
            .field(&self.vec.as_slice())
            .finish()
    }
}

impl<T, const N: usize> Drop for IntoIter<T, N> {
    fn drop(&mut self) {
        let len = self.vec.inner.len;
        if self.index < len {
            self.vec.refresh();
            unsafe {
                ptr::drop_in_place(core::slice::from_raw_parts_mut(
                    self.vec.inner.as_mut_ptr().add(self.index),
                    len - self.index,
                ));
                self.vec.inner.try_dealloc();
            }
        }
    }
}

mod tests {
    #[test]
    fn s() {}
}
