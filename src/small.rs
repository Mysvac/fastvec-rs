use alloc::alloc as malloc;
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::alloc::Layout;
use core::fmt::Debug;
use core::iter::FusedIterator;
use core::mem::{ManuallyDrop, MaybeUninit};
use core::num::NonZeroUsize;
use core::panic::{RefUnwindSafe, UnwindSafe};
use core::ptr::{self, NonNull};
use core::slice;

use super::utils::{IsZST, min_cap, split_range_bound};

// -----------------------------------------------------------------------------
// Utils

const MAX_LEN: usize = usize::MAX >> 1;
const MARKER: usize = usize::MAX ^ MAX_LEN;

union Data<T, const N: usize> {
    heap: (NonNull<T>, usize),
    cache: [ManuallyDrop<T>; N],
}

// -----------------------------------------------------------------------------
// SmallVec

/// A vector with inline storage that spills to heap when capacity is exceeded.
///
/// This type implements small-buffer optimization (SBO) while prioritizing the
/// space efficiency of the container itself:
/// - Storage mode (inline/heap) is encoded in a single bit of the length field.
/// - Heap metadata is stored in a compact union with the inline buffer.
///
/// Unlike `FastVec`, this type does not cache a data pointer, meaning all data
/// accesses require an additional branch to determine the storage location.
/// However, this design makes the container more compact. For example, `SmallVec<u64, 2>`
/// can have the same size as `Vec<u64>`.
///
/// Hot paths are optimized with cold path annotations to minimize the overhead
/// of branch checks, especially when data is inlined.
///
/// # Panics
/// Any operation that would cause `capacity > isize::MAX` will panic.
///
/// # Examples
///
/// ```
/// use fastvec::SmallVec;
///
/// // Allocate inline space for 2 elements (uninitialized)
/// let mut vec: SmallVec<String, 2> = SmallVec::new();
///
/// assert_eq!(vec.len(), 0);
/// assert_eq!(vec.capacity(), 2);
///
/// // Use it like a normal `Vec`
/// vec.push("Hello".to_string());
/// vec.push(", world!".to_string());
///
/// assert_eq!(vec, ["Hello", ", world!"]);
///
/// // Convert into a standard `Vec`
/// let vec: Vec<String> = vec.into_vec();
/// ```
pub struct SmallVec<T, const N: usize> {
    // The highest bit stores the location flag: 0 means cache, 1 means heap.
    // The remaining bits store length.
    len_and_flag: usize,
    data: Data<T, N>,
}

// -----------------------------------------------------------------------------
// Marker

unsafe impl<T: Sync, const N: usize> Sync for SmallVec<T, N> {}
unsafe impl<T: Send, const N: usize> Send for SmallVec<T, N> {}
impl<T, const N: usize> UnwindSafe for SmallVec<T, N> where T: UnwindSafe {}
impl<T, const N: usize> RefUnwindSafe for SmallVec<T, N> where T: RefUnwindSafe {}

// -----------------------------------------------------------------------------
// Basic

impl<T, const N: usize> Default for SmallVec<T, N> {
    /// Constructs a new, empty `SmallVec<T>`.
    ///
    /// # Panic
    /// Panic if the generic param `N` > `isize::MAX`.
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Drop for SmallVec<T, N> {
    fn drop(&mut self) {
        self.clear();
        self.dealloc();
    }
}

impl<T, const N: usize> SmallVec<T, N> {
    #[inline(always)]
    const fn in_heap(&self) -> bool {
        self.len_and_flag & MARKER != 0
    }

    /// Returns current capacity.
    ///
    /// Before spilling to heap this equals `N`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<i32, 2> = SmallVec::new();
    /// assert_eq!(vec.capacity(), 2);
    /// vec.extend([1, 2, 3]);
    /// assert!(vec.capacity() >= 3);
    /// ```
    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        if self.in_heap() {
            crate::utils::cold_path();
            unsafe { self.data.heap.1 }
        } else {
            N
        }
    }

    /// Returns the number of initialized elements in the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<i32, 2> = SmallVec::new();
    /// assert_eq!(vec.len(), 0);
    /// vec.push(1);
    /// assert_eq!(vec.len(), 1);
    /// ```
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len_and_flag & MAX_LEN
    }

    /// Returns `true` if the vector contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<i32, 2> = SmallVec::new();
    /// assert!(vec.is_empty());
    /// vec.push(1);
    /// assert!(!vec.is_empty());
    /// ```
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len_and_flag & MAX_LEN == 0
    }

    /// Returns a raw pointer to the vector's buffer, or a dangling raw
    /// pointer valid for zero sized reads if the vector didn't allocate.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<i32, 2> = SmallVec::new();
    /// vec.push(7);
    /// let ptr = vec.as_ptr();
    /// assert_eq!(unsafe { *ptr }, 7);
    /// ```
    #[inline(always)]
    pub const fn as_ptr(&self) -> *const T {
        if self.in_heap() {
            crate::utils::cold_path();
            unsafe { self.data.heap.0.as_ptr() }
        } else {
            unsafe { self.data.cache.as_ptr() as *const T }
        }
    }

    /// Returns a raw mutable pointer to the vector's buffer, or a dangling
    /// raw pointer valid for zero sized reads if the vector didn't allocate.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<i32, 2> = SmallVec::from([1, 2]);
    /// let ptr = vec.as_mut_ptr();
    /// unsafe { *ptr.add(1) = 9; }
    /// assert_eq!(vec.as_slice(), &[1, 9]);
    /// ```
    #[inline(always)]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        if self.in_heap() {
            crate::utils::cold_path();
            unsafe { self.data.heap.0.as_ptr() }
        } else {
            unsafe { self.data.cache.as_mut_ptr() as *mut T }
        }
    }

    /// Extracts a slice containing the entire vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let vec: SmallVec<_, 4> = [1, 2, 3].into();
    /// assert_eq!(vec.as_slice(), &[1, 2, 3]);
    /// ```
    #[inline]
    pub const fn as_slice(&self) -> &[T] {
        let data = self.as_ptr();
        let len = self.len();
        unsafe { slice::from_raw_parts(data, len) }
    }

    /// Extracts a mutable slice of the entire vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<_, 4> = [1, 2, 3].into();
    /// vec.as_mut_slice()[1] = 9;
    /// assert_eq!(vec.as_slice(), &[1, 9, 3]);
    /// ```
    #[inline]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        let data = self.as_mut_ptr();
        let len = self.len();
        unsafe { slice::from_raw_parts_mut(data, len) }
    }

    /// Forces the length of the vector to `new_len`.
    ///
    /// This is a low-level operation that does not initialize or drop elements.
    ///
    /// # Safety
    /// - `new_len <= capacity()`.
    /// - Elements in the range `old_len..new_len` must already be initialized.
    /// - Elements in the range `new_len..old_len` are considered logically removed
    ///   and will not be dropped by this call.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<i32, 4> = SmallVec::new();
    /// unsafe {
    ///     let ptr = vec.as_mut_ptr();
    ///     ptr.write(10);
    ///     ptr.add(1).write(20);
    ///     vec.set_len(2);
    /// }
    /// assert_eq!(vec.as_slice(), &[10, 20]);
    /// ```
    #[inline(always)]
    pub const unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.capacity());
        self.len_and_flag = (self.len_and_flag & MARKER) | new_len;
    }

    /// Clears the vector, removing all values.
    ///
    /// Note that this method has no effect on allocated capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<_, 2> = [1, 2, 3].into();
    /// let cap = vec.capacity();
    /// vec.clear();
    /// assert!(vec.is_empty());
    /// assert_eq!(vec.capacity(), cap);
    /// ```
    pub fn clear(&mut self) {
        if core::mem::needs_drop::<T>() {
            unsafe {
                let slice: &mut [T] = self.as_mut_slice();
                ptr::drop_in_place::<[T]>(slice);
            }
        }
        self.len_and_flag &= MARKER;
    }

    /// Shortens the vector, keeping the first `len` elements
    /// and dropping the rest.
    ///
    /// If `len >= self.len()`, this has no effect.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<_, 4> = [1, 2, 3, 4].into();
    /// vec.truncate(2);
    /// assert_eq!(vec.as_slice(), &[1, 2]);
    /// ```
    pub fn truncate(&mut self, len: usize) {
        let old = self.len_and_flag;
        let old_len = old & MAX_LEN;

        if old_len > len {
            if core::mem::needs_drop::<T>() {
                unsafe {
                    let data = self.as_mut_ptr().add(len);
                    let len = old_len - len;
                    let to_drop = ptr::slice_from_raw_parts_mut(data, len);
                    ptr::drop_in_place::<[T]>(to_drop)
                }
            }

            self.len_and_flag = (old & MARKER) | len;
        }
    }
}

// -----------------------------------------------------------------------------
// Memory

impl<T, const N: usize> SmallVec<T, N> {
    #[inline(never)]
    fn realloc(&mut self, cap: usize) {
        let len = self.len();

        debug_assert!(cap >= len);
        assert!(
            cap <= MAX_LEN,
            "the capacity of SmallVec cannot exceed isize::MAX"
        );

        if cap <= N {
            debug_assert!(self.in_heap());
            if !T::IS_ZST {
                let (ptr, old_cap) = unsafe { self.data.heap };
                let old_layout = Layout::array::<T>(old_cap).unwrap();
                self.len_and_flag = 0; // Ensure the safety during panic
                unsafe {
                    let dst = self.data.cache.as_mut_ptr() as *mut T;
                    ptr::copy_nonoverlapping::<T>(ptr.as_ptr(), dst, len);
                    malloc::dealloc(ptr.as_ptr() as *mut u8, old_layout);
                }
            }
            self.len_and_flag = len & MAX_LEN;
        } else if self.in_heap() {
            let (mut ptr, old_cap) = unsafe { self.data.heap };
            if !T::IS_ZST {
                let old_layout = Layout::array::<T>(old_cap).unwrap();
                let new_layout = Layout::array::<T>(cap).unwrap();
                let new_size = new_layout.size();
                let raw_ptr = ptr.as_ptr() as *mut u8;
                unsafe {
                    ptr = NonNull::new(malloc::realloc(raw_ptr, old_layout, new_size) as *mut T)
                        .unwrap_or_else(|| malloc::handle_alloc_error(new_layout));
                }
            }
            self.data.heap = (ptr, cap);
        } else {
            let ptr: NonNull<T> = if !T::IS_ZST {
                let layout = Layout::array::<T>(cap).unwrap();
                NonNull::new(unsafe { malloc::alloc(layout) as *mut T })
                    .unwrap_or_else(|| malloc::handle_alloc_error(layout))
            } else {
                let align = ::core::mem::align_of::<T>();
                debug_assert!(align != 0);
                NonNull::<T>::without_provenance(unsafe { NonZeroUsize::new_unchecked(align) })
            };
            if !T::IS_ZST {
                unsafe {
                    let src = self.data.cache.as_ptr() as *const T;
                    ptr::copy_nonoverlapping::<T>(src, ptr.as_ptr(), len);
                }
            }
            self.len_and_flag = len | MARKER;
            self.data.heap = (ptr, cap);
        }
    }

    fn dealloc(&mut self) {
        if !T::IS_ZST && self.in_heap() {
            let (ptr, cap) = unsafe { self.data.heap };
            let layout = Layout::array::<T>(cap).unwrap();
            unsafe {
                malloc::dealloc(ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

impl<T, const N: usize> SmallVec<T, N> {
    /// Constructs a new, empty `SmallVec<T>`.
    ///
    /// The vector will not allocate until the number of elements exceed `N`.
    ///
    /// # Panic
    /// Panic if the generic param `N` > `isize::MAX`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<i32, 4> = SmallVec::new();
    /// vec.push(1);
    /// assert_eq!(vec.as_slice(), &[1]);
    /// ```
    #[must_use]
    #[inline(always)]
    pub const fn new() -> Self {
        // This expressions can be eliminated at compile time.
        assert!(
            N <= MAX_LEN,
            "The capacity of SmallVec can not exceed isize::MAX"
        );

        Self {
            data: Data {
                // SAFETY: Full buffer uninitialized to internal uninitialized is safe.
                #[expect(clippy::uninit_assumed_init)]
                cache: unsafe { MaybeUninit::<[ManuallyDrop<T>; N]>::uninit().assume_init() },
            },
            len_and_flag: 0,
        }
    }

    /// Constructs a new, empty `SmallVec` with at least the specified capacity.
    ///
    /// If the specified capacity is less than or equal to `N`, this is equivalent
    /// to [`new`](SmallVec::new), and no heap memory will be allocated.
    ///
    /// # Panics
    /// Panics if `capacity > isize::MAX` elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let vec: SmallVec<i32, 4> = SmallVec::with_capacity(16);
    /// assert!(vec.capacity() >= 16);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        let mut this = Self::new();

        if capacity > N {
            this.realloc(capacity);
        }
        this
    }

    /// Creates a `SmallVec<T>` directly from a pointer, a length, and a capacity.
    ///
    /// The data is still stored using pointers and will not be moved.
    ///
    /// # Safety
    /// - `ptr` must be allocated with the global allocator for `capacity` elements of `T`.
    /// - `length <= capacity`.
    /// - The first `length` elements at `ptr` must be initialized.
    /// - The allocation must be uniquely owned and valid for deallocation by `SmallVec`.
    /// - `capacity` and `length` must both be `<= isize::MAX`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut source = vec![1, 2, 3];
    /// let ptr = source.as_mut_ptr();
    /// let (len, cap) = (source.len(), source.capacity());
    /// core::mem::forget(source);
    ///
    /// let small = unsafe { SmallVec::<i32, 2>::from_raw_parts(ptr, len, cap) };
    /// assert_eq!(small.as_slice(), &[1, 2, 3]);
    /// ```
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *mut T, length: usize, capacity: usize) -> Self {
        debug_assert!(length <= capacity && capacity & MARKER == 0);
        let ptr = NonNull::new(ptr).unwrap();

        Self {
            data: Data {
                heap: (ptr, capacity),
            },
            len_and_flag: length | MARKER,
        }
    }

    /// Creates a `SmallVec` by copying all elements from a slice.
    ///
    /// If `slice.len() <= N`, data is stored inline, otherwise it is allocated on heap.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let vec = SmallVec::<i32, 2>::from_slice(&[10, 20, 30]);
    /// assert_eq!(vec.as_slice(), &[10, 20, 30]);
    /// ```
    #[inline]
    pub fn from_slice(slice: &[T]) -> Self
    where
        T: Copy,
    {
        let mut this = Self::with_capacity(slice.len());
        unsafe {
            if !T::IS_ZST {
                let ptr = this.as_mut_ptr();
                ptr::copy_nonoverlapping(slice.as_ptr(), ptr, slice.len());
            }
            this.set_len(slice.len());
        }
        this
    }

    /// Reserves capacity for at least `additional`
    /// more elements to be inserted in the given `SmallVec<T>`.
    ///
    /// This may reserve more than requested to reduce future reallocations.
    ///
    /// # Panics
    /// Panics if the new capacity exceeds `isize::MAX`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<i32, 2> = SmallVec::new();
    /// vec.reserve(10);
    /// assert!(vec.capacity() >= 10);
    /// ```
    pub fn reserve(&mut self, additional: usize) {
        let cap = self.capacity();
        let len = self.len();
        let target = len.saturating_add(additional);
        if target > cap {
            self.realloc(target.min(MARKER).next_power_of_two());
        }
    }

    /// Reserves the minimum capacity for exactly `additional` more elements.
    ///
    /// Unlike [`reserve`](Self::reserve), this does not intentionally over-allocate.
    ///
    /// # Panics
    /// Panics if the new capacity exceeds `isize::MAX`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<i32, 2> = SmallVec::new();
    /// vec.reserve_exact(5);
    /// assert!(vec.capacity() >= 5);
    /// ```
    pub fn reserve_exact(&mut self, additional: usize) {
        let cap = self.capacity();
        let len = self.len();
        let target = len.saturating_add(additional);
        if target > cap {
            self.realloc(target);
        }
    }

    /// Shrinks the capacity of the vector as much as possible.
    ///
    /// If `len <= N`, data may move back to inline storage.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<_, 2> = SmallVec::with_capacity(16);
    /// vec.extend([1, 2, 3]);
    /// vec.shrink_to_fit();
    /// assert!(vec.capacity() >= vec.len());
    /// ```
    pub fn shrink_to_fit(&mut self) {
        if self.in_heap() {
            let len = self.len();
            let cap = self.capacity();
            if cap > len {
                self.realloc(len);
            }
        }
    }

    /// Shrinks the capacity of the vector as much as possible.
    ///
    /// The resulting capacity will be at least `max(self.len(), min_capacity)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<_, 2> = SmallVec::with_capacity(32);
    /// vec.extend([1, 2, 3]);
    /// vec.shrink_to(4);
    /// assert!(vec.capacity() >= 4);
    /// ```
    pub fn shrink_to(&mut self, min_capacity: usize) {
        if self.in_heap() {
            let len = self.len();
            let cap = self.capacity();
            if min_capacity >= len && min_capacity < cap {
                self.realloc(min_capacity);
            }
        }
    }

    /// Converts the `SmallVec` into `Vec<T>`.
    ///
    /// If the data is already in the heap, transfer pointer directly.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let small: SmallVec<_, 2> = [1, 2, 3].into();
    /// let vec = small.into_vec();
    /// assert_eq!(vec, vec![1, 2, 3]);
    /// ```
    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        if self.in_heap() {
            let (ptr, cap) = unsafe { self.data.heap };
            let len = self.len();
            ::core::mem::forget(self);
            unsafe { Vec::from_raw_parts(ptr.as_ptr(), len, cap) }
        } else {
            // !in_heap, self.len_and_flag == self.len()
            let len = self.len_and_flag;
            let mut ret = Vec::<T>::with_capacity(len);
            unsafe {
                if !T::IS_ZST {
                    let src = self.data.cache.as_ptr() as *const T;
                    let dst = ret.as_mut_ptr();
                    ptr::copy_nonoverlapping(src, dst, len);
                }

                ::core::mem::forget(self);
                ret.set_len(len);
            }
            ret
        }
    }

    /// Converts the `SmallVec` into `Box<[T]>`.
    ///
    /// If the data is already in the heap, transfer pointer directly.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let small: SmallVec<_, 2> = [1, 2, 3].into();
    /// let boxed = small.into_boxed_slice();
    /// assert_eq!(&*boxed, &[1, 2, 3]);
    /// ```
    #[inline]
    pub fn into_boxed_slice(self) -> Box<[T]> {
        self.into_vec().into_boxed_slice()
    }

    /// Appends an element to the back of a collection.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<_, 2> = SmallVec::new();
    /// vec.push(1);
    /// vec.push(2);
    /// vec.push(3);
    /// assert_eq!(vec.as_slice(), &[1, 2, 3]);
    /// ```
    pub fn push(&mut self, value: T) {
        if self.in_heap() {
            crate::utils::cold_path();
            let len = self.len();
            let cap = unsafe { self.data.heap.1 };

            if len == cap {
                crate::utils::cold_path();
                let new_cap = (cap << 1).max(min_cap::<T>());
                self.realloc(new_cap);
            }
            unsafe {
                let ptr = self.data.heap.0.as_ptr();
                ptr::write(ptr.add(len), value);
                self.len_and_flag += 1;
            }
        } else {
            // !in_heap, self.len_and_flag == self.len()
            let len = self.len_and_flag;
            let ptr: *mut T = if len == N {
                crate::utils::cold_path();
                let new_cap = (N << 1).max(min_cap::<T>());
                self.realloc(new_cap);
                unsafe { self.data.heap.0.as_ptr() }
            } else {
                unsafe { self.data.cache.as_mut_ptr() as *mut T }
            };
            unsafe {
                ptr::write(ptr.add(len), value);
            }
            self.len_and_flag += 1;
        }
    }

    /// Appends an element to the back of the vector without checking capacity.
    ///
    /// # Safety
    /// `self.len() < self.capacity()` must hold before calling this method.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<i32, 2> = SmallVec::new();
    /// unsafe {
    ///     vec.push_unchecked(1);
    ///     vec.push_unchecked(2);
    /// }
    /// assert_eq!(vec.as_slice(), &[1, 2]);
    /// ```
    #[inline(always)]
    pub unsafe fn push_unchecked(&mut self, value: T) {
        let ptr = self.as_mut_ptr();
        let len = self.len();
        unsafe {
            ptr::write(ptr.add(len), value);
        }
        self.len_and_flag += 1;
    }

    /// Removes the last element and returns it, or `None` if empty.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<_, 2> = [1, 2].into();
    /// assert_eq!(vec.pop(), Some(2));
    /// assert_eq!(vec.pop(), Some(1));
    /// assert_eq!(vec.pop(), None);
    /// ```
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        let len = self.len();
        if len != 0 {
            unsafe {
                self.len_and_flag -= 1;
                Some(ptr::read(self.as_ptr().add(len - 1)))
            }
        } else {
            crate::utils::cold_path();
            None
        }
    }

    /// Removes and returns the last element if `predicate` returns `true`.
    ///
    /// Returns `None` when the vector is empty or predicate returns `false`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<_, 4> = [1, 2, 3, 4].into();
    /// assert_eq!(vec.pop_if(|v| *v % 2 == 0), Some(4));
    /// assert_eq!(vec.pop_if(|v| *v % 2 == 0), None);
    /// ```
    #[inline]
    pub fn pop_if(&mut self, predicate: impl FnOnce(&mut T) -> bool) -> Option<T> {
        let last = self.last_mut()?;
        if predicate(last) { self.pop() } else { None }
    }

    /// Inserts `element` at `index`, shifting all elements after it to the right.
    ///
    /// # Panics
    /// Panics if `index > len`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<_, 2> = [1, 3].into();
    /// vec.insert(1, 2);
    /// assert_eq!(vec.as_slice(), &[1, 2, 3]);
    /// ```
    pub fn insert(&mut self, index: usize, element: T) {
        #[cold]
        #[inline(never)]
        fn assert_failed(index: usize, len: usize) -> ! {
            panic!("insertion index (is {index}) should be <= len (is {len})");
        }

        let len = self.len();
        if index > len {
            assert_failed(index, len);
        }

        // space for the new element
        if len == self.capacity() {
            crate::utils::cold_path();
            self.reserve(1);
        }

        unsafe {
            let p = self.as_mut_ptr().add(index);
            if index < len {
                ptr::copy(p, p.add(1), len - index);
            }
            ptr::write(p, element);
            self.len_and_flag += 1;
        }
    }

    /// Removes and returns the element at `index`, shifting later elements left.
    ///
    /// # Panics
    /// Panics if `index >= len`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<_, 4> = [10, 20, 30].into();
    /// assert_eq!(vec.remove(1), 20);
    /// assert_eq!(vec.as_slice(), &[10, 30]);
    /// ```
    pub fn remove(&mut self, index: usize) -> T {
        #[cold]
        #[inline(never)]
        fn assert_failed(index: usize, len: usize) -> ! {
            panic!("removal index (is {index}) should be < len (is {len})");
        }

        let len = self.len();
        if index >= len {
            assert_failed(index, len);
        }

        unsafe {
            let ptr = self.as_mut_ptr().add(index);
            let ret = ptr::read(ptr);
            ptr::copy(ptr.add(1), ptr, len - index - 1);
            self.len_and_flag -= 1;
            ret
        }
    }

    /// Removes and returns the element at `index`.
    ///
    /// The last element is moved into `index`, so ordering is not preserved.
    ///
    /// # Panics
    /// Panics if `index >= len`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<_, 4> = [10, 20, 30, 40].into();
    /// let removed = vec.swap_remove(1);
    /// assert_eq!(removed, 20);
    /// assert_eq!(vec.len(), 3);
    /// ```
    pub fn swap_remove(&mut self, index: usize) -> T {
        #[cold]
        #[inline(never)]
        fn assert_failed(index: usize, len: usize) -> ! {
            panic!("swap_remove index (is {index}) should be < len (is {len})");
        }

        let len = self.len();
        if index >= len {
            assert_failed(index, len);
        }

        unsafe {
            let ptr = self.as_mut_ptr();
            let value = ptr::read(ptr.add(index));
            ptr::copy(ptr.add(len - 1), ptr.add(index), 1);
            self.len_and_flag -= 1;
            value
        }
    }

    /// Moves all elements from `other` to the end of `self`.
    ///
    /// `other` is emptied after the call.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut a: SmallVec<_, 2> = [1, 2].into();
    /// let mut b: SmallVec<_, 4> = [3, 4].into();
    /// a.append(&mut b);
    /// assert_eq!(a.as_slice(), &[1, 2, 3, 4]);
    /// assert!(b.is_empty());
    /// ```
    pub fn append<const P: usize>(&mut self, other: &mut SmallVec<T, P>) {
        unsafe {
            let slice = other.as_slice();
            let count = slice.len();
            self.reserve(count);

            if !T::IS_ZST {
                let len = self.len();
                let dst = self.as_mut_ptr().add(len);
                ptr::copy_nonoverlapping::<T>(slice.as_ptr(), dst, count);
            }

            self.len_and_flag += count;
            other.set_len(0);
        }
    }

    /// Resizes the vector to `new_len` using `f` to generate new values.
    ///
    /// If `new_len < len`, this truncates the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<_, 2> = [1].into();
    /// vec.resize_with(4, || 7);
    /// assert_eq!(vec.as_slice(), &[1, 7, 7, 7]);
    /// ```
    pub fn resize_with<F>(&mut self, new_len: usize, mut f: F)
    where
        F: FnMut() -> T,
    {
        let len = self.len();
        if new_len > len {
            self.reserve(new_len - len);
            let ptr = self.as_mut_ptr();
            (len..new_len).for_each(|idx| unsafe {
                ptr::write(ptr.add(idx), f());
            });
            unsafe {
                self.set_len(new_len);
            }
        } else {
            self.truncate(new_len);
        }
    }

    /// Retains only elements for which `f` returns `true`, passing each item mutably.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<_, 4> = [1, 2, 3, 4].into();
    /// vec.retain_mut(|v| {
    ///     *v *= 2;
    ///     *v > 4
    /// });
    /// assert_eq!(vec.as_slice(), &[6, 8]);
    /// ```
    pub fn retain_mut<F: FnMut(&mut T) -> bool>(&mut self, mut f: F) {
        let base_ptr = self.as_mut_ptr();
        let len = self.len_and_flag & MAX_LEN;
        let flag = self.len_and_flag & MARKER;
        self.len_and_flag = 0;
        let mut count = 0usize;

        for index in 0..len {
            unsafe {
                let dst = base_ptr.add(index);
                if f(&mut *dst) {
                    ptr::copy(dst, base_ptr.add(count), 1);
                    count += 1;
                } else {
                    ptr::drop_in_place(dst);
                }
            }
        }
        self.len_and_flag = count | flag;
    }

    /// Retains only elements for which `f` returns `true`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<_, 4> = [1, 2, 3, 4].into();
    /// vec.retain(|v| *v % 2 == 0);
    /// assert_eq!(vec.as_slice(), &[2, 4]);
    /// ```
    #[inline]
    pub fn retain<F: FnMut(&T) -> bool>(&mut self, mut f: F) {
        self.retain_mut(|v| f(v));
    }

    /// Removes consecutive repeated elements according to `same_bucket`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<_, 8> = [10, 20, 21, 30, 20].into();
    /// vec.dedup_by(|a, b| *a / 10 == *b / 10);
    /// assert_eq!(vec.as_slice(), &[10, 20, 30, 20]);
    /// ```
    pub fn dedup_by<F: FnMut(&mut T, &mut T) -> bool>(&mut self, mut same_bucket: F) {
        let len = self.len();
        if len <= 1 {
            return;
        }

        let ptr = self.as_mut_ptr();
        let mut left = 0usize;

        unsafe {
            let mut p_l = ptr.add(left);
            for right in 1..len {
                let p_r = ptr.add(right);
                if !same_bucket(&mut *p_r, &mut *p_l) {
                    left += 1;
                    p_l = ptr.add(left);
                    if right != left {
                        ptr::swap(p_r, p_l);
                    }
                }
            }
        }
        self.truncate(left + 1);
    }

    /// Removes consecutive repeated elements that map to the same key.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<_, 8> = [10, 20, 21, 30, 20].into();
    /// vec.dedup_by_key(|v| *v / 10);
    /// assert_eq!(vec.as_slice(), &[10, 20, 30, 20]);
    /// ```
    #[inline]
    pub fn dedup_by_key<F, K>(&mut self, mut key: F)
    where
        F: FnMut(&mut T) -> K,
        K: PartialEq,
    {
        self.dedup_by(|a, b| key(a) == key(b));
    }

    /// Returns the remaining spare capacity as a slice of `MaybeUninit<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use core::mem::MaybeUninit;
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<i32, 4> = SmallVec::new();
    /// let spare: &mut [MaybeUninit<i32>] = vec.spare_capacity_mut();
    /// spare[0].write(11);
    /// unsafe { vec.set_len(1); }
    /// assert_eq!(vec.as_slice(), &[11]);
    /// ```
    #[inline]
    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        let len = self.len();
        let cap = self.capacity();
        unsafe {
            let data = self.as_mut_ptr().add(len);
            slice::from_raw_parts_mut(data as *mut MaybeUninit<T>, cap - len)
        }
    }
}

// -----------------------------------------------------------------------------
// Common

super::utils::impl_common_traits!(SmallVec<T, N>);

impl<T, U, const N: usize> PartialEq<SmallVec<U, N>> for SmallVec<T, N>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &SmallVec<U, N>) -> bool {
        PartialEq::eq(self.as_slice(), other.as_slice())
    }
}

impl<T: PartialEq, const N: usize> SmallVec<T, N> {
    /// Removes consecutive repeated elements using `PartialEq`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<_, 8> = [1, 1, 2, 2, 3].into();
    /// vec.dedup();
    /// assert_eq!(vec.as_slice(), &[1, 2, 3]);
    /// ```
    #[inline]
    pub fn dedup(&mut self) {
        self.dedup_by(|x, y| PartialEq::eq(x, y));
    }
}

impl<T: Clone, const N: usize> Clone for SmallVec<T, N> {
    fn clone(&self) -> Self {
        let slice = self.as_slice();

        let mut this = Self::with_capacity(slice.len());
        slice.iter().for_each(|item| unsafe {
            this.push_unchecked(item.clone());
        });
        this
    }

    fn clone_from(&mut self, source: &Self) {
        let slice = source.as_slice();
        self.clear();

        self.reserve(slice.len());
        slice.iter().for_each(|item| unsafe {
            self.push_unchecked(item.clone());
        });
    }
}

impl<T: Clone, const N: usize> SmallVec<T, N> {
    /// Resizes the vector to `new_len` by cloning `value` when growing.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<_, 2> = [1, 2].into();
    /// vec.resize(4, 9);
    /// assert_eq!(vec.as_slice(), &[1, 2, 9, 9]);
    /// vec.resize(1, 0);
    /// assert_eq!(vec.as_slice(), &[1]);
    /// ```
    pub fn resize(&mut self, new_len: usize, value: T) {
        let len = self.len();

        if new_len > len {
            self.reserve(new_len - len);
            (len..new_len - 1).for_each(|_| unsafe {
                self.push_unchecked(value.clone());
            });
            unsafe {
                self.push_unchecked(value);
            }
        } else {
            self.truncate(new_len);
        }
    }

    /// Extends the vector by cloning all elements from `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut vec: SmallVec<_, 2> = [1].into();
    /// vec.extend_from_slice(&[2, 3, 4]);
    /// assert_eq!(vec.as_slice(), &[1, 2, 3, 4]);
    /// ```
    pub fn extend_from_slice(&mut self, other: &[T]) {
        self.reserve(other.len());
        other.iter().for_each(|item| unsafe {
            self.push_unchecked(item.clone());
        });
    }
}

impl<'a, T: 'a + Clone + 'a, const N: usize> Extend<&'a T> for SmallVec<T, N> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);

        iter.for_each(|item| {
            self.push(item.clone());
        });
    }
}

impl<T, const N: usize> Extend<T> for SmallVec<T, N> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);

        iter.for_each(|item| {
            self.push(item);
        });
    }
}

impl<T, const N: usize> FromIterator<T> for SmallVec<T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut vec = Self::with_capacity(iter.size_hint().0);
        iter.for_each(|item| {
            vec.push(item);
        });
        vec
    }
}

// -----------------------------------------------------------------------------
// From/Into

impl<T, const N: usize> From<SmallVec<T, N>> for Vec<T> {
    fn from(v: SmallVec<T, N>) -> Self {
        v.into_vec()
    }
}

impl<T, const N: usize> From<SmallVec<T, N>> for Box<[T]> {
    fn from(v: SmallVec<T, N>) -> Self {
        v.into_boxed_slice()
    }
}

impl<T, const N: usize, const P: usize> TryFrom<SmallVec<T, N>> for [T; P] {
    type Error = SmallVec<T, N>;

    fn try_from(mut vec: SmallVec<T, N>) -> Result<[T; P], SmallVec<T, N>> {
        if vec.len() != P {
            return Err(vec);
        }

        let src = vec.as_ptr();
        unsafe { vec.set_len(0) };
        let array = unsafe { ptr::read(src as *const [T; P]) };
        Ok(array)
    }
}

impl<T: Clone, const N: usize> From<&[T]> for SmallVec<T, N> {
    fn from(s: &[T]) -> SmallVec<T, N> {
        let mut vec = SmallVec::<T, N>::with_capacity(s.len());
        s.iter().for_each(|item| unsafe {
            vec.push_unchecked(item.clone());
        });
        vec
    }
}

impl<T: Clone, const N: usize> From<&mut [T]> for SmallVec<T, N> {
    fn from(s: &mut [T]) -> SmallVec<T, N> {
        let mut vec = SmallVec::<T, N>::with_capacity(s.len());
        s.iter().for_each(|item| unsafe {
            vec.push_unchecked(item.clone());
        });
        vec
    }
}

impl<T: Clone, const N: usize, const P: usize> From<&[T; N]> for SmallVec<T, P> {
    fn from(s: &[T; N]) -> SmallVec<T, P> {
        Self::from(s.as_slice())
    }
}

impl<T: Clone, const N: usize, const P: usize> From<&mut [T; N]> for SmallVec<T, P> {
    fn from(s: &mut [T; N]) -> Self {
        Self::from(s.as_mut_slice())
    }
}

impl<T, const N: usize, const P: usize> From<[T; N]> for SmallVec<T, P> {
    fn from(s: [T; N]) -> Self {
        if N <= P {
            let mut this = Self::new();
            let ptr = unsafe { this.data.cache.as_mut_ptr() as *mut T };
            let s = ManuallyDrop::new(s);
            let len = s.len();
            unsafe {
                ptr::copy_nonoverlapping(s.as_ptr(), ptr, len);
                this.set_len(len);
            }
            this
        } else {
            let vec = Vec::<T>::from(s);
            SmallVec::from(vec)
        }
    }
}

impl<T, const N: usize> From<Vec<T>> for SmallVec<T, N> {
    fn from(s: Vec<T>) -> Self {
        let (p, l, c) = s.into_raw_parts();
        unsafe { SmallVec::from_raw_parts(p, l, c) }
    }
}

impl<T, const N: usize> From<Box<[T]>> for SmallVec<T, N> {
    fn from(s: Box<[T]>) -> Self {
        Self::from(s.into_vec())
    }
}

// -----------------------------------------------------------------------------
// IntoIterator

/// An iterator that consumes a [`SmallVec`] and yields its items by value.
///
/// # Examples
///
/// ```
/// # use fastvec::SmallVec;
///
/// let vec = SmallVec::<_, 5>::from(["1", "2", "3"]);
/// let mut iter = vec.into_iter();
///
/// assert_eq!(iter.next(), Some("1"));
///
/// let vec: Vec<&'static str> = iter.collect();
/// assert_eq!(vec, ["2", "3"]);
/// ```
pub struct IntoIter<T, const N: usize> {
    vec: ManuallyDrop<SmallVec<T, N>>,
    index: usize,
}

unsafe impl<T: Send, const N: usize> Send for IntoIter<T, N> {}
unsafe impl<T: Sync, const N: usize> Sync for IntoIter<T, N> {}

impl<T, const N: usize> IntoIterator for SmallVec<T, N> {
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
        if self.index < self.vec.len() {
            self.index += 1;
            unsafe { Some(ptr::read(self.vec.as_ptr().add(self.index - 1))) }
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let v = self.vec.len() - self.index;
        (v, Some(v))
    }
}

impl<T, const N: usize> DoubleEndedIterator for IntoIter<T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let len = self.vec.len();
        if self.index < len {
            unsafe {
                self.vec.set_len(len - 1);
            }
            unsafe { Some(ptr::read(self.vec.as_ptr().add(len - 1))) }
        } else {
            None
        }
    }
}

impl<T, const N: usize> ExactSizeIterator for IntoIter<T, N> {
    #[inline]
    fn len(&self) -> usize {
        self.vec.len() - self.index
    }
}

impl<T, const N: usize> FusedIterator for IntoIter<T, N> {}

impl<T: Debug, const N: usize> Debug for IntoIter<T, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("IntoIter")
            .field(&self.vec.as_slice())
            .finish()
    }
}

impl<T, const N: usize> Drop for IntoIter<T, N> {
    fn drop(&mut self) {
        let len = self.vec.len();
        if self.index < len {
            unsafe {
                let ptr = self.vec.as_mut_ptr().add(self.index);
                let count = len - self.index;
                let to_drop = ptr::slice_from_raw_parts_mut(ptr, count);
                ptr::drop_in_place(to_drop);
            }
        }
        self.vec.dealloc();
    }
}

// -----------------------------------------------------------------------------
// Drain

/// An iterator that removes the items from a [`SmallVec`]
/// and yields them by value.
///
/// See [`SmallVec::drain`] .
pub struct Drain<'a, T: 'a, const N: usize> {
    tail_start: usize,
    tail_len: usize,
    iter: slice::Iter<'a, T>,
    vec: NonNull<SmallVec<T, N>>,
}

impl<T, const N: usize> SmallVec<T, N> {
    /// Removes the subslice indicated by the given range from the vector,
    /// returning a double-ended iterator over the removed subslice.
    ///
    /// If the iterator is dropped before being fully consumed, it drops the remaining removed elements.
    ///
    /// The returned iterator keeps a mutable borrow on the vector to optimize its implementation.
    ///
    /// See more information in [`Vec::drain`].
    ///
    /// # Panics
    ///
    /// Panics if the range is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut v = SmallVec::<_, 3>::from([1, 2, 3]);
    /// let u: Vec<_> = v.drain(1..).collect();
    /// assert_eq!(v, [1]);
    /// assert_eq!(u, [2, 3]);
    ///
    /// // A full range clears the vector, like `clear()` does
    /// v.drain(..);
    /// assert_eq!(v, []);
    /// ```
    pub fn drain<R: core::ops::RangeBounds<usize>>(&mut self, range: R) -> Drain<'_, T, N> {
        let len = self.len();
        let (start, end) = split_range_bound(&range, len);

        unsafe {
            self.set_len(start);
            let data = self.as_ptr().add(start);
            let range_slice = slice::from_raw_parts(data, end - start);

            Drain {
                tail_start: end,
                tail_len: len - end,
                iter: range_slice.iter(),
                vec: NonNull::new_unchecked(self as *mut _),
            }
        }
    }
}

impl<T: Debug, const N: usize> Debug for Drain<'_, T, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Drain").field(&self.iter.as_slice()).finish()
    }
}

impl<T, const N: usize> Iterator for Drain<'_, T, N> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.iter
            .next()
            .map(|reference| unsafe { ptr::read(reference) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<T, const N: usize> DoubleEndedIterator for Drain<'_, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter
            .next_back()
            .map(|reference| unsafe { ptr::read(reference) })
    }
}

impl<T, const N: usize> ExactSizeIterator for Drain<'_, T, N> {
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<T, const N: usize> FusedIterator for Drain<'_, T, N> {}

impl<'a, T: 'a, const N: usize> Drop for Drain<'a, T, N> {
    fn drop(&mut self) {
        /// Moves back the un-`Drain`ed elements to restore the original `Vec`.
        struct DropGuard<'r, 'a, T, const N: usize>(&'r mut Drain<'a, T, N>);

        impl<'r, 'a, T, const N: usize> Drop for DropGuard<'r, 'a, T, N> {
            fn drop(&mut self) {
                if self.0.tail_len > 0 {
                    unsafe {
                        let source_vec = self.0.vec.as_mut();
                        // memmove back untouched tail, update to new length
                        let start = source_vec.len();
                        let tail = self.0.tail_start;
                        if tail != start {
                            let base = source_vec.as_mut_ptr();
                            let src = base.add(tail);
                            let dst = base.add(start);
                            ptr::copy(src, dst, self.0.tail_len);
                        }
                        source_vec.set_len(start + self.0.tail_len);
                    }
                }
            }
        }

        let iter = core::mem::take(&mut self.iter);
        let drop_len = iter.len();

        let mut vec = self.vec;

        if T::IS_ZST {
            // ZSTs have no identity, so we don't need to move them around, we only need to drop the correct amount.
            // this can be achieved by manipulating the Vec length instead of moving values out from `iter`.
            unsafe {
                let vec = vec.as_mut();
                let old_len = vec.len();
                vec.set_len(old_len + drop_len + self.tail_len);
                vec.truncate(old_len + self.tail_len);
            }

            return;
        }

        // ensure elements are moved back into their appropriate places, even when drop_in_place panics
        let _guard = DropGuard(self);

        if drop_len == 0 {
            return;
        }

        // as_slice() must only be called when iter.len() is > 0 because
        // it also gets touched by vec::Splice which may turn it into a dangling pointer
        // which would make it and the vec pointer point to different allocations which would
        // lead to invalid pointer arithmetic below.
        let drop_ptr = iter.as_slice().as_ptr();

        unsafe {
            // drop_ptr comes from a slice::Iter which only gives us a &[T] but for drop_in_place
            // a pointer with mutable provenance is necessary. Therefore we must reconstruct
            // it from the original vec but also avoid creating a &mut to the front since that could
            // invalidate raw pointers to it which some unsafe code might rely on.
            let vec_ptr = vec.as_mut().as_mut_ptr();
            let drop_offset = drop_ptr.offset_from_unsigned(vec_ptr);
            let to_drop = ptr::slice_from_raw_parts_mut(vec_ptr.add(drop_offset), drop_len);
            ptr::drop_in_place(to_drop);
        }
    }
}

// -----------------------------------------------------------------------------
// ExtractIf

/// An iterator that removes elements matching a predicate from a range.
///
/// This yields removed items by value while compacting retained elements in place.
///
/// See [`SmallVec::extract_if`] .
pub struct ExtractIf<'a, T, F: FnMut(&mut T) -> bool, const N: usize> {
    vec: &'a mut SmallVec<T, N>,
    idx: usize,
    end: usize,
    del: usize,
    old_len: usize,
    pred: F,
}

impl<T, const N: usize> SmallVec<T, N> {
    /// Creates an iterator which uses a closure to determine
    /// if an element in the range should be removed.
    ///
    /// See more information in [`Vec::extract_if`].
    ///
    /// # Panics
    /// Panics if the range is out of bounds.
    ///
    ///
    /// # Examples
    ///
    /// Splitting a vector into even and odd values, reusing the original vector:
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut numbers = SmallVec::<_, 5>::from([1, 2, 3, 4, 5, 6, 8, 9, 11, 13, 14, 15]);
    ///
    /// let evens = numbers.extract_if(.., |x| *x % 2 == 0).collect::<SmallVec<_, 10>>();
    /// let odds = numbers;
    ///
    /// assert_eq!(evens, [2, 4, 6, 8, 14]);
    /// assert_eq!(odds, [1, 3, 5, 9, 11, 13, 15]);
    /// ```
    ///
    /// Using the range argument to only process a part of the vector:
    ///
    /// ```
    /// # use fastvec::SmallVec;
    /// let mut items = SmallVec::<_, 5>::from([0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2]);
    /// let ones = items.extract_if(7.., |x| *x == 1).collect::<Vec<_>>();
    /// assert_eq!(items, [0, 0, 0, 0, 0, 0, 0, 2, 2, 2]);
    /// assert_eq!(ones.len(), 3);
    /// ```
    pub fn extract_if<F, R>(&mut self, range: R, filter: F) -> ExtractIf<'_, T, F, N>
    where
        F: FnMut(&mut T) -> bool,
        R: core::ops::RangeBounds<usize>,
    {
        let old_len = self.len();
        let (start, end) = split_range_bound(&range, old_len);

        // Guard against the vec getting leaked (leak amplification)
        unsafe {
            self.set_len(0);
        }

        ExtractIf {
            vec: self,
            idx: start,
            del: 0,
            end,
            old_len,
            pred: filter,
        }
    }
}

impl<T, F: FnMut(&mut T) -> bool, const N: usize> Iterator for ExtractIf<'_, T, F, N> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        while self.idx < self.end {
            let i = self.idx;
            // SAFETY:
            //  We know that `i < self.end` from the if guard and that `self.end <= self.old_len` from
            //  the validity of `Self`. Therefore `i` points to an element within `vec`.
            //
            //  Additionally, the i-th element is valid because each element is visited at most once
            //  and it is the first time we access vec[i].
            //
            //  Note: we can't use `vec.get_unchecked_mut(i)` here since the precondition for that
            //  function is that i < vec.len(), but we've set vec's length to zero.
            let cur = unsafe { &mut *self.vec.as_mut_ptr().add(i) };
            let drained = (self.pred)(cur);
            // Update the index *after* the predicate is called. If the index
            // is updated prior and the predicate panics, the element at this
            // index would be leaked.
            self.idx += 1;
            if drained {
                self.del += 1;
                // SAFETY: We never touch this element again after returning it.
                return Some(unsafe { ptr::read(cur) });
            } else if self.del > 0 {
                // SAFETY: `self.del` > 0, so the hole slot must not overlap with current element.
                // We use copy for move, and never touch this element again.
                unsafe {
                    let hole_slot = self.vec.as_mut_ptr().add(i - self.del);
                    ptr::copy_nonoverlapping(cur, hole_slot, 1);
                }
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.end - self.idx))
    }
}

impl<T, F: FnMut(&mut T) -> bool, const N: usize> Drop for ExtractIf<'_, T, F, N> {
    fn drop(&mut self) {
        if !T::IS_ZST && self.del > 0 {
            // SAFETY: Trailing unchecked items must be valid since we never touch them.
            unsafe {
                let base = self.vec.as_mut_ptr();
                ptr::copy(
                    base.add(self.idx),
                    base.add(self.idx - self.del),
                    self.old_len - self.idx,
                );
            }
        }
        // SAFETY: After filling holes, all items are in contiguous memory.
        unsafe {
            self.vec.set_len(self.old_len - self.del);
        }
    }
}

impl<T: Debug, F: FnMut(&mut T) -> bool, const N: usize> Debug for ExtractIf<'_, T, F, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let peek = if self.idx < self.end {
            self.vec.get(self.idx)
        } else {
            None
        };
        f.debug_struct("ExtractIf")
            .field("peek", &peek)
            .finish_non_exhaustive()
    }
}

// -----------------------------------------------------------------------------
// Tests

#[cfg(test)]
mod tests {
    use super::SmallVec;
    use core::sync::atomic::{AtomicUsize, Ordering};

    macro_rules! define_tracker {
        () => {
            static DROPS: AtomicUsize = AtomicUsize::new(0);

            struct Tracker;
            impl Drop for Tracker {
                fn drop(&mut self) {
                    DROPS.fetch_add(1, Ordering::SeqCst);
                }
            }
        };
    }

    #[test]
    fn drop_zst() {
        define_tracker!();

        DROPS.store(0, Ordering::SeqCst);

        let mut vec = SmallVec::<Tracker, 0>::new();
        vec.push(Tracker);
        vec.push(Tracker);
        vec.push(Tracker);
        vec.push(Tracker);
        vec.push(Tracker);

        {
            let mut drain = vec.drain(1..4);
            let one = drain.next_back().unwrap();
            drop(one);
            assert_eq!(DROPS.load(Ordering::SeqCst), 1);
        }

        assert_eq!(DROPS.load(Ordering::SeqCst), 3);

        drop(vec);
        assert_eq!(DROPS.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn drop_inline_and_heap() {
        define_tracker!();

        DROPS.store(0, Ordering::SeqCst);
        {
            let mut vec = SmallVec::<Tracker, 4>::new();
            vec.push(Tracker);
            vec.push(Tracker);
            vec.push(Tracker);
            assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        }
        assert_eq!(DROPS.load(Ordering::SeqCst), 3);

        DROPS.store(0, Ordering::SeqCst);
        {
            let mut vec = SmallVec::<Tracker, 2>::new();
            vec.push(Tracker);
            vec.push(Tracker);
            vec.push(Tracker);
            assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        }
        assert_eq!(DROPS.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn drop_pop_remove() {
        define_tracker!();

        DROPS.store(0, Ordering::SeqCst);

        let mut vec = SmallVec::<Tracker, 2>::new();
        vec.push(Tracker);
        vec.push(Tracker);
        vec.push(Tracker);

        let popped = vec.pop().unwrap();
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        drop(popped);
        assert_eq!(DROPS.load(Ordering::SeqCst), 1);

        let removed = vec.remove(0);
        assert_eq!(DROPS.load(Ordering::SeqCst), 1);
        drop(removed);
        assert_eq!(DROPS.load(Ordering::SeqCst), 2);

        drop(vec);
        assert_eq!(DROPS.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn drop_into_iter() {
        define_tracker!();

        DROPS.store(0, Ordering::SeqCst);

        let mut vec = SmallVec::<Tracker, 2>::new();
        vec.push(Tracker);
        vec.push(Tracker);
        vec.push(Tracker);

        let mut iter = vec.into_iter();
        let first = iter.next().unwrap();
        drop(first);
        assert_eq!(DROPS.load(Ordering::SeqCst), 1);

        drop(iter);
        assert_eq!(DROPS.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn drop_drain() {
        define_tracker!();

        DROPS.store(0, Ordering::SeqCst);

        let mut vec = SmallVec::<Tracker, 2>::new();
        vec.push(Tracker);
        vec.push(Tracker);
        vec.push(Tracker);
        vec.push(Tracker);
        vec.push(Tracker);

        {
            let mut drain = vec.drain(1..4);
            let first = drain.next().unwrap();
            drop(first);
            assert_eq!(DROPS.load(Ordering::SeqCst), 1);
        }

        // 1 consumed + 2 still in drained range
        assert_eq!(DROPS.load(Ordering::SeqCst), 3);

        drop(vec);
        assert_eq!(DROPS.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn drop_extract_if() {
        static DROPS: AtomicUsize = AtomicUsize::new(0);

        struct Tracker {
            id: usize,
        }
        impl Drop for Tracker {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }

        DROPS.store(0, Ordering::SeqCst);

        let mut vec = SmallVec::<Tracker, 2>::new();
        for id in 0..6 {
            vec.push(Tracker { id });
        }

        let removed: SmallVec<Tracker, 8> = vec.extract_if(.., |t| t.id % 2 == 0).collect();
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);

        drop(removed);
        assert_eq!(DROPS.load(Ordering::SeqCst), 3);

        drop(vec);
        assert_eq!(DROPS.load(Ordering::SeqCst), 6);
    }
}
