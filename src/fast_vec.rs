use alloc::{boxed::Box, vec::Vec};
use core::{
    fmt,
    iter::FusedIterator,
    marker::PhantomData,
    mem::{self, ManuallyDrop},
    panic::RefUnwindSafe,
    ptr,
};

pub use crate::fast_vec_data::{Drain, ExtractIf, FastVecData, Splice};
use crate::utils::{IsZST, cold_path};

/// A stack-prioritized vector that automatically spills to the heap
/// when capacity is exceeded.
///
/// Unlike [`SmallVec`](https://docs.rs/smallvec/latest/smallvec/),
/// [`FastVec`] uses **pointer caching** to avoid conditional checks
/// on every operation, achieving higher performance.
///
/// When the data is in the stack area, the execution efficiency is
/// almost the same as `[T; N]`. Even if switching to the heap, it
/// won't be slower than [`Vec`].
///
/// But the cost is that this type is [`!Sync`](Sync) and requires
/// operate through [`FastVecData`].
///
/// So the real advantage of [`FastVec`] lies in data processing rather
/// than storage, and it is usually recommended to convert it to [`Vec`]
/// when transferring data.
///
/// If [`FastVec`]'s data is already in the heap, this conversion only
/// requires copying pointers, which is very cheap. If it is on the stack,
/// it is equivalent to only applying for heap memory once, won't be more
/// expensive than using [`Vec`].
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
/// assert!(vec.in_stack()); // Data is on stack
///
/// // If requested capacity exceeds N, data goes to heap
/// let mut vec = FastVec::<i32, 2>::with_capacity(4);
/// assert!(!vec.in_stack()); // Data is on heap
/// ```
///
/// ## Modifying Data
///
/// Most data-modifying operations require obtaining a
/// [`&mut FastVecData`](FastVecData) via [`FastVec::data`].
///
/// ```
/// # use fastvec::{FastVec, fast_vec::FastVecData};
/// let mut vec: FastVec<_> = [1, 2, 3, 4].into();
/// let data: &mut FastVecData<_,_> = vec.data();
///
/// // Use it like a Vec
/// data.push(5);
/// data.insert(0, 6);
///
/// assert_eq!(data, &[6, 1, 2, 3, 4, 5]);
/// ```
///
/// # API Design
///
/// [`FastVec`] supports nearly all [`Vec`] methods, categorized as follows:
///
/// ## Operations Through [`&mut FastVecData`](FastVecData)
///
/// Operations that take `&self` or `&mut self` require:
/// - [`push`](FastVecData::push), [`pop`](FastVecData::pop)
/// - [`insert`](FastVecData::insert), [`remove`](FastVecData::remove)
/// - [`drain`](FastVecData::drain), [`extract_if`](FastVecData::extract_if)
/// - And more...
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
/// - [`len`](FastVec::len), [`capacity`](FastVec::capacity), [`is_empty`](FastVec::is_empty);
///   they have no additional expenses.
/// - [`as_slice`](FastVec::as_slice), [`as_mut_slice`](FastVec::as_mut_slice);
///   they internally call [`data`](FastVec::data) first.
///
/// ## Trait Implementations
///
/// [`FastVec`] implements [`Deref`](core::ops::Deref), [`Index`](core::ops::Index),
/// [`Debug`](core::fmt::Debug), etc., via [`as_slice`](FastVec::as_slice) and
/// [`as_mut_slice`](FastVec::as_mut_slice):
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
/// **Performance note:** These operations call `get` each time.
/// For complex operations like `sort`, this overhead is negligible.
/// However, for simple operations (`Index`, `push`, `pop`), the overhead is measurable.
///
/// ## Recommended Usage Pattern
///
/// For best performance, acquire the data reference once and reuse it:
///
/// ```
/// # use fastvec::FastVec;
/// let mut vec: FastVec<_> = [1, 4, 3, 2].into();
/// let data = vec.data();
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
/// # Understanding `FastVecData`
///
/// ## The Problem
///
/// A naive stack-to-heap vector looks like this:
///
/// ```ignore
/// struct NaiveVec<T, const N: usize> {
///     stack_cache: [MaybeUninit<T>; N],
///     heap_ptr: *mut T,
///     len: usize,
///     cap: usize,
///     in_stack: bool, // Is data on stack or heap?
/// }
/// ```
///
/// **Problem:** Every operation (`push`, `pop`, `index`, etc.) must check `in_stack` to determine
/// whether to access `stack_cache` or `heap_ptr`. This conditional is cheap individually but cumulative
/// overhead becomes significant for simple operations.
///
/// ## The Solution: Pointer Caching
///
/// Make a single pointer always point to the current data location:
///
/// ```ignore
/// struct FastVecData<T, const N: usize> {
///     stack_cache: [MaybeUninit<T>; N],
///     ptr: Cell<*mut T>,  // Always points to active data
///     len: usize,
///     cap: usize,
///     in_stack: bool,     // Only checked during reallocation
/// }
/// ```
///
/// Now `ptr` directly accesses data without branching. The `in_stack` check is only needed when
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
/// When you call [`data`](FastVec::data), [`FastVec`]:
/// 1. Refreshes the pointer (if data is on stack)
/// 2. Returns a borrow of [`FastVecData`]
///
/// Rust's borrow checker ensures [`FastVecData`] cannot be moved while borrowed, so the pointer
/// remains valid during handle usage.
///
/// ## Why [`Cell`](core::cell::Cell)?
///
/// Pointer refresh needs interior mutability (even [`as_slice`](FastVec::as_slice) must update the pointer).
///
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
/// - The generic parameter `N` remains semantically meaningful: [`in_stack`](FastVec::in_stack)
///   and [`capacity`](FastVec::capacity) behave as if space were allocated
///
/// This ensures API consistency across all types while maintaining zero overhead for ZSTs.
///
/// # Thread Safety
///
/// **[`FastVec`]**: Implements [`Send`] but **not** [`Sync`] due to internal
/// [`Cell`](core::cell::Cell) usage (required for pointer relocation). Concurrent
/// calls to [`as_slice`](FastVec::as_slice) may race.
///
/// - **[`FastVecData`]**: Implements both [`Send`] and [`Sync`], so you can safely
///   share its reference across threads.
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
impl<T, const N: usize> RefUnwindSafe for FastVec<T, N> where T: RefUnwindSafe {}

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
    /// # use fastvec::{FastVec, fastvec};
    /// let vec: FastVec<i32, 8> = FastVec::new();
    ///
    /// // equivalent to this
    /// let vec: FastVec<i32> = fastvec![];
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
    /// assert!(vec.in_stack());
    ///
    /// let vec: FastVec<i32, 5> = FastVec::with_capacity(10);
    /// assert!(!vec.in_stack());
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
    /// # Safety
    /// - if T is **not** zero sized type, **capacity > 0**.
    ///
    /// See more information in [`Vec::from_raw_parts`].
    #[inline]
    pub const unsafe fn from_raw_parts(ptr: *mut T, length: usize, capacity: usize) -> Self {
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
    /// vec.data().push(1);
    /// assert!(!vec.is_empty());
    /// ```
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
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
    /// vec.data().extend([1, 2, 3,  4, 5]);
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
    /// vec.data().extend([1, 2, 3,  4, 5]);
    /// assert!(vec.capacity() >= 9);
    /// assert_eq!(vec.len(), 9);
    /// ```
    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        self.inner.cap
    }

    /// Returns `true` if the data is stored in the stack cache.
    ///
    /// It's possible that `capacity` or `len` <= `N` but the data is not in stack.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<i32, 8> = [1, 2, 3, 4].into();
    /// assert!(vec.in_stack());
    ///
    /// vec.data().extend([1, 2, 3,  4, 5]);
    /// assert!(!vec.in_stack());
    /// ```
    #[inline(always)]
    pub const fn in_stack(&self) -> bool {
        self.inner.in_stack
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

    /// Refresh the pointer and return a mutable reference to the internal data.
    ///
    /// You can use this mutable reference for methods such as `push`, `pop`, `retain`, and `insert`.
    /// The pointer is refreshed once in `data`; later method calls reuse it without extra cost.
    ///
    /// We do not provide a version for obtaining immutable borrowing,
    /// you can use [`as_slice`](FastVec::as_slice) instead.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<i32, 8> = [1, 2, 3, 4].into();
    /// let v = vec.data();
    ///
    /// v.push(5);
    /// v.retain(|x| *x % 2 == 1);
    ///
    /// assert_eq!(vec, [1, 3, 5]);
    /// ```
    #[inline]
    pub fn data(&mut self) -> &mut FastVecData<T, N> {
        self.refresh();
        &mut self.inner
    }

    /// Refresh the pointer and obtain slices of the data.
    ///
    /// During the validity period of the slice reference, the data will not be moved, so the pointer is valid.
    ///
    /// This method enables [`FastVec`] to implement many traits directly through slice access.
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
    ///
    /// A better approach is to obtain a reference once and then use it multiple times.
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<i32, 8> = [1, 4, 3, 2].into();
    /// let slice = vec.as_slice();
    ///
    /// let mut x = vec[1];
    /// x += vec[2] * vec[3];
    /// ```
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
    ///
    /// A better approach is to obtain a reference once and then use it multiple times.
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<i32, 8> = [1, 4, 3, 2].into();
    /// let slice = vec.as_mut_slice();
    ///
    /// slice.sort();
    ///
    /// let mut x = vec[1];
    /// x += vec[2] * vec[3];
    /// ```
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.refresh();
        self.inner.as_mut_slice()
    }

    /// Convert [`FastVec`] to [`Vec`].
    ///
    /// - If the data is in the stack, the exact memory will be allocated.
    /// - If the data is already on the heap, no reallocation is needed.
    ///
    /// The returned [`Vec`] may not be tight because heap data does not shrink here.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let vec: FastVec<i32, 3> = [1, 2].into();
    /// assert!(vec.in_stack());
    /// let vec: Vec<_> = vec.into_vec();
    /// assert_eq!(vec, [1, 2]);
    /// assert!(vec.capacity() == 2);
    ///
    /// let vec: FastVec<i32, 3> = [1, 2, 3, 4, 5].into();
    /// assert!(!vec.in_stack());
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
    /// - If the data is in the stack, the exact memory will be allocated.
    /// - If the data is in the heap, [`Vec::shrink_to_fit`] will be called.
    pub fn shrink_into_vec(self) -> Vec<T> {
        self.refresh();
        self.inner.shrink_into_vec()
    }

    /// Convert [`FastVec`] to [`Box<[T]>`](Box).
    pub fn into_boxed_slice(self) -> Box<[T]> {
        self.refresh();
        self.inner.into_boxed_slice()
    }

    /// Convert [`FastVec`] to a leaked slice.
    ///
    /// This will first move the data to the heap to
    /// ensure that the returned references are valid.
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
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let vec: FastVec<i32, 3> = [1, 2].into();
    /// let mut vec: FastVec<i32, 5> = vec.force_cast();
    /// assert_eq!(vec, [1, 2]);
    /// assert!(vec.in_stack());
    ///
    /// vec.data().extend([3, 4, 5, 6]);
    /// assert!(!vec.in_stack());
    ///
    /// let vec: FastVec<i32, 8> = vec.force_cast();
    /// assert_eq!(vec, [1, 2, 3, 4, 5, 6]);
    /// assert!(!vec.in_stack());
    /// ```
    pub fn force_cast<const P: usize>(mut self) -> FastVec<T, P> {
        if self.inner.in_stack {
            let len = self.inner.len;
            let mut state = <FastVec<T, P>>::with_capacity(len);
            let dst = state.data();
            let src = self.data();
            unsafe {
                if !T::IS_ZST {
                    ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), len);
                }
                dst.len = len;
                src.len = 0;
            }
            state
        } else {
            unsafe {
                let state = <FastVec<T, P>>::from_raw_parts(
                    self.data().as_mut_ptr(),
                    self.len(),
                    self.capacity(),
                );
                mem::forget(self);
                state
            }
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
            let vec = state.data();
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
    /// assert_eq!(vec.data().pop(), Some([7, 8, 9]));
    /// assert!(!vec.in_stack());
    ///
    /// let mut flattened = vec.into_flattened::<6>();
    /// assert_eq!(flattened, [1, 2, 3, 4, 5, 6]);
    /// assert!(!flattened.in_stack());
    ///
    /// let mut vec: FastVec<_, 3> = [[1, 2, 3], [4, 5, 6], [7, 8, 9]].into();
    /// assert_eq!(vec.data().pop(), Some([7, 8, 9]));
    /// assert!(vec.in_stack());
    ///
    /// let mut flattened = vec.into_flattened::<5>();
    /// assert_eq!(flattened, [1, 2, 3, 4, 5, 6]);
    /// assert!(!flattened.in_stack());
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
    /// See [`Clone::clone`]
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{FastVec, fastvec};
    /// let vec: FastVec<i32> = fastvec![1, 2 , 3];
    ///
    /// let vec2 = vec.clone();
    /// assert_eq!(vec, [1, 2 , 3]);
    /// assert_eq!(vec, vec2);
    /// ```
    fn clone(&self) -> Self {
        let mut vec = Self::with_capacity(self.len());
        let dst = vec.data();
        for item in self.as_slice() {
            unsafe {
                dst.push_unchecked(item.clone());
            }
        }
        vec
    }

    /// See [`Clone::clone_from`]
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{FastVec, fastvec};
    /// let vec: FastVec<i32> = fastvec![1, 2 , 3];
    ///
    /// let mut vec2 = fastvec![];
    /// vec2.clone_from(&vec);
    /// assert_eq!(vec, [1, 2 , 3]);
    /// assert_eq!(vec, vec2);
    /// ```
    fn clone_from(&mut self, source: &Self) {
        let dst = self.data();
        dst.clear();
        dst.reserve(source.len());

        for item in source.as_slice() {
            unsafe {
                dst.push_unchecked(item.clone());
            }
        }
    }
}

impl<T: Clone, const N: usize> From<&[T]> for FastVec<T, N> {
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec = <FastVec<i32, 3>>::from([1, 2, 3].as_slice());
    ///
    /// assert_eq!(vec, [1, 2, 3]);
    /// assert!(vec.in_stack());
    /// ```
    fn from(value: &[T]) -> Self {
        let mut res = Self::with_capacity(value.len());
        let vec = res.data();
        for items in value {
            unsafe {
                vec.push_unchecked(items.clone());
            }
        }
        res
    }
}

impl<T: Clone, const N: usize, const P: usize> From<&[T; P]> for FastVec<T, N> {
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec = <FastVec<i32, 3>>::from(&[1, 2, 3]);
    ///
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    fn from(value: &[T; P]) -> Self {
        let mut res = Self::with_capacity(value.len());
        let vec = res.data();
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

impl<T, const N: usize, const P: usize> From<[T; P]> for FastVec<T, N> {
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec = <FastVec<i32, 3>>::from([1, 2, 3]);
    ///
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    fn from(value: [T; P]) -> Self {
        let mut vec = Self::with_capacity(P);
        let vec_mut = vec.data();
        unsafe {
            ptr::copy_nonoverlapping(value.as_ptr(), vec_mut.as_mut_ptr(), P);
            vec_mut.len = P;
            mem::forget(value);
        }
        vec
    }
}

impl<T, const N: usize> From<Box<[T]>> for FastVec<T, N> {
    /// This is efficient because it directly moves the pointer.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec = <FastVec<i32, 3>>::from(vec![1, 2, 3].into_boxed_slice());
    ///
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    #[inline]
    fn from(value: Box<[T]>) -> Self {
        Self::from(value.into_vec())
    }
}

impl<T, const N: usize> From<Vec<T>> for FastVec<T, N> {
    /// This is efficient because it directly moves the pointer.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec = <FastVec<i32, 3>>::from(vec![1, 2, 3]);
    ///
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    fn from(mut value: Vec<T>) -> Self {
        let capacity = value.capacity();
        let length = value.len();

        if capacity == 0 && length == 0 {
            cold_path();
            Self::new()
        } else {
            let ptr = value.as_mut_ptr();
            mem::forget(value);
            unsafe { Self::from_raw_parts(ptr, length, capacity) }
        }
    }
}

impl<T, const N: usize, const P: usize> From<crate::StackVec<T, P>> for FastVec<T, N> {
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{StackVec, FastVec};
    /// let mut vec = <FastVec<i32, 3>>::from(
    ///     StackVec::<i32, 3>::from([1, 2, 3])
    /// );
    ///
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    fn from(mut value: crate::StackVec<T, P>) -> Self {
        let len = value.len();
        let mut vec = Self::with_capacity(len);
        let vec_mut = vec.data();
        unsafe {
            ptr::copy_nonoverlapping(value.as_ptr(), vec_mut.as_mut_ptr(), len);
            vec_mut.len = len;
            value.set_len(0);
        }
        vec
    }
}

impl<T, const N: usize, const P: usize> From<crate::AutoVec<T, P>> for FastVec<T, N> {
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{FastVec, AutoVec};
    /// let mut vec = <FastVec<i32>>::from(
    ///     AutoVec::<i32, 3>::from([1, 2, 3])
    /// );
    ///
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    fn from(value: crate::AutoVec<T, P>) -> Self {
        match value.0 {
            crate::auto_vec::InnerVec::Stack(stack_vec) => Self::from(stack_vec),
            crate::auto_vec::InnerVec::Heap(items) => Self::from(items),
        }
    }
}

impl<T, const N: usize> FromIterator<T> for FastVec<T, N> {
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec = <FastVec<i32, 3>>::from_iter([1, 2, 3].into_iter());
    ///
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (hint, _) = iter.size_hint();
        let mut res = Self::with_capacity(hint);
        let vec = res.data();
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
            self.vec.inner.len = len - 1;
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
                ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                    self.vec.inner.as_mut_ptr().add(self.index),
                    len - self.index,
                ));
                self.vec.inner.try_dealloc();
            }
        }
    }
}
