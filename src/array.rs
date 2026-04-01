use alloc::boxed::Box;
use alloc::vec::Vec;
use core::fmt::Debug;
use core::iter::FusedIterator;
use core::mem::{ManuallyDrop, MaybeUninit};
use core::panic::{RefUnwindSafe, UnwindSafe};
use core::ptr::NonNull;
use core::{ptr, slice};

use super::utils::{IsZST, split_range_bound, zst_init};

// -----------------------------------------------------------------------------
// ArrayVec

/// A vector with a fixed capacity.
///
/// The vector is a contiguous value (storing the elements inline) that you can store
/// in any location (stack, heap, static), while its elements always live inline
/// inside the container.
///
/// It mirrors most of the API of [`Vec`], but maintains the same high efficiency as `[T; N]`.
///
/// # Panics
/// Any operation that causes `len > capacity`.
///
/// # Examples
///
/// ```
/// use fastvec::ArrayVec;
///
/// // Allocate uninitialized inline space for 10 elements
/// let mut vec: ArrayVec<String, 10> = ArrayVec::new();
///
/// assert_eq!(vec.len(), 0);
/// assert_eq!(vec.capacity(), 10);
///
/// // Then you can use it like `Vec`, the only difference is that
/// // the capacity is fixed.
/// vec.push("Hello".to_string());
/// vec.push(", world!".to_string());
///
/// assert_eq!(vec, ["Hello", ", world!"]);
///
/// // Convert into `Vec` to transfer ownership across scopes.
/// let vec: Vec<String> = vec.into_vec();
/// // There is only one heap allocation in the entire process.
/// ```
///
/// # ZST support
///
/// For zero sized types, this data will not allocate additional space,
/// Therefore, corresponding to ZST, the capacity can be set very large.
/// (e.g. `ArrayVec<(), usize::MAX>`).
pub struct ArrayVec<T, const N: usize> {
    len: usize,
    data: [MaybeUninit<T>; N],
}

// -----------------------------------------------------------------------------
// Marker

unsafe impl<T: Sync, const N: usize> Sync for ArrayVec<T, N> {}
unsafe impl<T: Send, const N: usize> Send for ArrayVec<T, N> {}
impl<T, const N: usize> UnwindSafe for ArrayVec<T, N> where T: UnwindSafe {}
impl<T, const N: usize> RefUnwindSafe for ArrayVec<T, N> where T: RefUnwindSafe {}

// -----------------------------------------------------------------------------
// Basic

impl<T, const N: usize> Default for ArrayVec<T, N> {
    /// Constructs a new, empty `ArrayVec` with inline storage
    /// with the specified capacity.
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Drop for ArrayVec<T, N> {
    fn drop(&mut self) {
        if core::mem::needs_drop::<T>() {
            self.clear();
        }
    }
}

impl<T, const N: usize> ArrayVec<T, N> {
    /// Returns the maximum number of elements the vector can hold.
    ///
    /// The capacity is fixed at compile time by the const generic parameter `N`.
    ///
    /// # Examples
    /// ```
    /// # use fastvec::ArrayVec;
    /// let vec = ArrayVec::<String, 5>::new();
    /// assert_eq!(vec.capacity(), 5);
    /// ```
    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Returns the number of elements in the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::ArrayVec;
    /// let vec = ArrayVec::<String, 5>::new();
    /// assert_eq!(vec.len(), 0);
    /// ```
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector contains no elements.
    ///
    /// # Examples
    /// ```
    /// # use fastvec::ArrayVec;
    /// let mut v = ArrayVec::<i32, 5>::new();
    /// assert!(v.is_empty());
    ///
    /// v.push(1);
    /// assert!(!v.is_empty());
    /// ```
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns `true` if `len == N` .
    ///
    /// # Examples
    /// ```
    /// # use fastvec::ArrayVec;
    /// let mut v = ArrayVec::<i32, 3>::new();
    /// assert!(!v.is_full());
    ///
    /// v.extend([1, 2, 3]);
    /// assert!(v.is_full());
    /// ```
    #[inline(always)]
    pub const fn is_full(&self) -> bool {
        self.len == N
    }

    /// Returns a raw pointer to the vector’s buffer, or a dangling pointer
    /// valid for zero-sized reads if `T` is a zero-sized type.
    ///
    /// The caller must ensure that the vector outlives the pointer this function returns,
    /// or else it will end up dangling.
    ///
    /// Modifying the vector will **not** cause its buffer to be reallocated.
    /// However, moving 'ArrayVec' itself will make ptr invalid.
    #[inline(always)]
    pub const fn as_ptr(&self) -> *const T {
        &raw const self.data as *const T
    }

    /// Returns a raw mutable pointer to the vector’s buffer, or a dangling pointer
    /// valid for zero-sized reads if `T` is a zero-sized type.
    ///
    /// The caller must ensure that the vector outlives the pointer this function returns,
    /// or else it will end up dangling.
    ///
    /// Modifying the vector will **not** cause its buffer to be reallocated.
    /// However, moving 'ArrayVec' itself will make ptr invalid.
    #[inline(always)]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        &raw mut self.data as *mut T
    }

    /// Extracts a slice containing the entire vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::ArrayVec;
    /// let mut vec = ArrayVec::from([1, 2, 3, 4]);
    /// assert_eq!(vec.as_slice(), &[1, 2, 3, 4]);
    ///
    /// vec.pop();
    /// assert_eq!(vec.as_slice(), &[1, 2, 3]);
    /// ```
    #[inline]
    pub const fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len) }
    }

    /// Extracts a mutable slice containing the entire vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::ArrayVec;
    /// let mut vec = ArrayVec::from([1, 2, 3, 4]);
    /// let slice = vec.as_mut_slice();
    ///
    /// slice[3] = 5;
    /// assert_eq!(vec, [1, 2, 3, 5]);
    /// ```
    #[inline]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len) }
    }

    /// Forces the length of the vector to `new_len`.
    ///
    /// This is a low-level operation that maintains none of the normal invariants of the type.
    ///
    /// # Safety
    /// - `new_len` needs to be less than or equal to capacity `N`.
    /// - If the length is increased, it is necessary to ensure that the new element is initialized correctly.
    /// - If the length is reduced, it is necessary to ensure that the reduced elements can be dropped normally.
    ///
    /// See more information in [`Vec::set_len`].
    #[inline(always)]
    pub const unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= N);
        self.len = new_len
    }

    /// Clears the vector, removing all values.
    ///
    /// # Examples
    /// ```
    /// # use fastvec::ArrayVec;
    /// let mut v: ArrayVec::<i32, 5> = ArrayVec::new();
    /// v.extend([1, 2, 3]);
    /// assert!(!v.is_empty());
    ///
    /// v.clear();
    /// assert!(v.is_empty());
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        if core::mem::needs_drop::<T>() {
            unsafe {
                let slice: &mut [T] = self.as_mut_slice();
                ptr::drop_in_place::<[T]>(slice);
            }
        }
        self.len = 0;
    }

    /// Shortens the vector, keeping the first `len` elements
    /// and dropping the rest.
    ///
    /// If `len >= self.len()`, this has no effect.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::ArrayVec;
    /// let mut vec = ArrayVec::from([1, 2, 3, 4, 5]);
    /// let x = vec.truncate(2);
    /// assert_eq!(vec.len(), 2);
    /// ```
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        if self.len > len {
            if core::mem::needs_drop::<T>() {
                unsafe {
                    let data = self.as_mut_ptr().add(len);
                    let len = self.len - len;
                    let to_drop = ptr::slice_from_raw_parts_mut(data, len);
                    ptr::drop_in_place::<[T]>(to_drop);
                }
            }

            self.len = len;
        }
    }
}

// -----------------------------------------------------------------------------
// Memory

impl<T, const N: usize> ArrayVec<T, N> {
    /// Constructs a new, empty `ArrayVec` with inline storage and fixed capacity.
    ///
    /// The capacity must be provided at compile time via the const generic parameter.
    ///
    /// The inline buffer is embedded in the `ArrayVec` value itself.
    /// If the value is stack-allocated, very large capacities may still risk stack overflow.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::ArrayVec;
    /// let mut vec: ArrayVec<i32, 8> = ArrayVec::new();
    /// vec.push(1);
    /// vec.push(2);
    /// assert_eq!(vec, [1, 2]);
    /// ```
    #[must_use]
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            // SAFETY: Full buffer uninitialized to internal uninitialized is safe.
            data: unsafe { MaybeUninit::<[MaybeUninit<T>; N]>::uninit().assume_init() },
            len: 0,
        }
    }

    /// Copy data from a slice and create a `ArrayVec`.
    ///
    /// # Panics
    ///
    /// Panics if the length of slice exceed `N`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::ArrayVec;
    /// let vec = ArrayVec::<i32, 3>::from_slice(&[10, 20, 30]);
    /// assert_eq!(vec.as_slice(), &[10, 20, 30]);
    /// ```
    #[inline]
    #[must_use]
    pub const fn from_slice(slice: &[T]) -> Self
    where
        T: Copy,
    {
        let length = slice.len();
        assert!(length <= N, "length overflow");

        let mut vec = Self::new();
        if !T::IS_ZST {
            unsafe {
                let src = slice.as_ptr();
                let dst = vec.as_mut_ptr();
                ptr::copy_nonoverlapping(src, dst, length);
            }
        }

        vec.len = length;
        vec
    }

    /// Converts a `ArrayVec` to a `Vec`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::ArrayVec;
    /// let mut vec = ArrayVec::<String, 3>::new();
    /// vec.push("123".to_string());
    ///
    /// let vec = vec.into_vec();
    /// assert_eq!(vec.len(), 1);
    /// assert_eq!(vec.capacity(), 1);
    /// ```
    #[inline]
    pub fn into_vec(mut self) -> Vec<T> {
        let mut vec: Vec<T> = Vec::with_capacity(self.len);

        unsafe {
            ptr::copy_nonoverlapping(self.as_ptr(), vec.as_mut_ptr(), self.len);
            vec.set_len(self.len);
            self.len = 0;
        }

        vec
    }

    /// Converts the `ArrayVec` into `Box<[T]>`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::ArrayVec;
    /// let small: ArrayVec<_, 3> = [1, 2, 3].into();
    /// let boxed = small.into_boxed_slice();
    /// assert_eq!(&*boxed, &[1, 2, 3]);
    /// ```
    #[inline]
    pub fn into_boxed_slice(self) -> Box<[T]> {
        self.into_vec().into_boxed_slice()
    }

    /// Appends an element to the back of the vector.
    ///
    /// # Panics
    /// Panics if the vector is full (`len == N`).
    ///
    /// # Time complexity
    /// O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::ArrayVec;
    /// let mut vec = ArrayVec::<i32, 5>::new();
    /// vec.push(1);
    /// vec.push(2);
    /// assert_eq!(vec.len(), 2);
    /// ```
    #[inline]
    pub const fn push(&mut self, value: T) {
        assert!(!self.is_full(), "length overflow during `push`");
        unsafe {
            self.push_unchecked(value);
        }
    }

    /// Appends an element to the back of the vector.
    ///
    /// Return `Err(input)` if the vector is full.
    ///
    /// # Time complexity
    /// O(1)
    ///
    /// ```
    /// # use fastvec::ArrayVec;
    /// let mut array = ArrayVec::<i32, 2>::new();
    ///
    /// let push1 = array.try_push(1);
    /// let push2 = array.try_push(2);
    ///
    /// assert!(push1.is_ok());
    /// assert!(push2.is_ok());
    ///
    /// assert_eq!(&array, &[1, 2]);
    ///
    /// let overflow = array.try_push(3);
    /// assert_eq!(overflow, Err(3));
    /// ```
    #[inline]
    pub const fn try_push(&mut self, value: T) -> Result<(), T> {
        if self.is_full() {
            Err(value)
        } else {
            unsafe {
                self.push_unchecked(value);
            }
            Ok(())
        }
    }

    /// Appends an element to the back of the vector without bounds checking.
    ///
    /// # Safety
    /// length < capacity `N` (before `push`)
    #[inline(always)]
    pub const unsafe fn push_unchecked(&mut self, value: T) {
        let len: usize = self.len;

        if T::IS_ZST {
            ::core::mem::forget(value);
        } else {
            unsafe {
                ptr::write(self.as_mut_ptr().add(len), value);
            }
        }

        self.len = len + 1;
    }

    /// Removes an item from the end of the vector and
    /// returns it, or `None` if empty.
    ///
    /// # Time complexity
    /// O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::ArrayVec;
    /// let mut vec = ArrayVec::<i32, 5>::new();
    /// vec.push(1);
    /// let one = vec.pop().unwrap();
    ///
    /// assert_eq!(one, 1);
    /// assert_eq!(vec.len(), 0);
    /// assert_eq!(vec.pop(), None);
    /// ```
    #[inline(always)]
    pub const fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            crate::utils::cold_path();
            None
        } else {
            unsafe {
                self.len -= 1;
                // This hint is provided to the caller of the `pop`, not the `pop` itself.
                core::hint::assert_unchecked(self.len < self.capacity());
                if T::IS_ZST {
                    Some(zst_init())
                } else {
                    Some(ptr::read(self.as_ptr().add(self.len)))
                }
            }
        }
    }

    /// Removes and returns the last element from a vector if
    /// the predicate returns `true`, or `None` if the predicate
    /// returns false or the vector is empty (the predicate will
    /// not be called in that case).
    #[inline]
    pub fn pop_if(&mut self, predicate: impl FnOnce(&mut T) -> bool) -> Option<T> {
        if self.len == 0 {
            crate::utils::cold_path();
            None
        } else {
            unsafe {
                let ptr = self.as_mut_ptr().add(self.len - 1);
                if predicate(&mut *ptr) {
                    self.len -= 1;
                    Some(ptr::read(ptr))
                } else {
                    None
                }
            }
        }
    }

    /// Inserts an element at position `index` within the vector, shifting all
    /// elements after it to the right.
    ///
    /// # Panics
    /// Panic if the array is full or the `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::ArrayVec;
    /// let mut vec = ArrayVec::<i32, 5>::new();
    /// vec.insert(0, 1);
    /// vec.insert(0, 2);
    /// let two = vec.pop().unwrap();
    /// assert_eq!(two, 1);
    /// assert_eq!(vec.len(), 1);
    /// ```
    #[inline]
    pub const fn insert(&mut self, index: usize, element: T) {
        assert!(index <= self.len, "insertion index should be <= len");
        assert!(self.len < N, "length overflow during `insert`");

        if T::IS_ZST {
            ::core::mem::forget(element);
        } else {
            unsafe {
                let ptr = self.as_mut_ptr().add(index);
                if index < self.len {
                    ptr::copy(ptr, ptr.add(1), self.len - index);
                }
                ptr::write(ptr, element);
            }
        }

        self.len += 1;
    }

    /// Inserts an element at position `index` within the vector, shifting all
    /// elements after it to the right.
    ///
    /// Because this shifts over the remaining elements, it has a worst-case performance of O(n).
    /// If you don’t need the order of elements to be preserved, use swap_remove instead.
    ///
    /// # Panics
    ///
    /// Panics if `index >= len`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::ArrayVec;
    /// let mut vec = ArrayVec::from([1, 2, 3]);
    ///
    /// assert_eq!(vec.remove(0), 1);
    /// assert_eq!(vec, [2, 3]);
    /// ```
    #[inline]
    pub const fn remove(&mut self, index: usize) -> T {
        assert!(index < self.len, "removal index should be < len");

        unsafe {
            let value: T;

            if T::IS_ZST {
                value = zst_init();
            } else {
                let ptr = self.as_mut_ptr().add(index);
                value = ptr::read(ptr);
                ptr::copy(ptr.add(1), ptr, self.len - index - 1);
            }

            self.len -= 1;
            value
        }
    }

    /// Removes an element from the vector and returns it.
    ///
    /// The removed element is replaced by the last element of the vector.
    ///
    /// This does not preserve ordering of the remaining elements, but is O(1).
    /// If you need to preserve the element order, use [`remove`](ArrayVec::remove) instead.
    ///
    /// # Panics
    ///
    /// Panics if `index >= len`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::ArrayVec;
    /// let mut vec = ArrayVec::from([1, 2, 3]);
    ///
    /// assert_eq!(vec.swap_remove(0), 1);
    /// assert_eq!(vec, [3, 2]);
    ///
    /// assert_eq!(vec.swap_remove(1), 2);
    /// assert_eq!(vec, [3]);
    /// ```
    #[inline]
    pub const fn swap_remove(&mut self, index: usize) -> T {
        assert!(index < self.len, "removal index should be < len");

        unsafe {
            let value: T;

            if T::IS_ZST {
                value = zst_init();
            } else {
                let base_ptr = self.as_mut_ptr();
                value = ptr::read(base_ptr.add(index));
                ptr::copy(base_ptr.add(self.len - 1), base_ptr.add(index), 1);
            }

            self.len -= 1;
            value
        }
    }

    /// Moves all elements from `other` to the end of `self`.
    ///
    /// `other` is emptied after the call.
    ///
    /// # Panics
    /// Panics if the new length exceeds N.
    pub fn append<const P: usize>(&mut self, other: &mut ArrayVec<T, P>) {
        assert!(other.len <= N - self.len, "length overflow during `append`");

        unsafe {
            let slice = other.as_slice();
            let count = slice.len();

            if !T::IS_ZST {
                let len = self.len();
                let dst = self.as_mut_ptr().add(len);
                ptr::copy_nonoverlapping::<T>(slice.as_ptr(), dst, count);
            }

            self.len += count;
            other.len = 0;
        }
    }

    /// Resizes the `ArrayVec` in-place so that len is equal to new_len.
    ///
    /// # Panics
    /// Panics if the new length exceeds N.
    pub fn resize_with<F: FnMut() -> T>(&mut self, new_len: usize, mut f: F) {
        assert!(new_len <= N, "length overflow during `resize_with`");

        let len = self.len();
        if new_len > len {
            let ptr = self.as_mut_ptr();
            (len..new_len).for_each(|idx| unsafe {
                ptr::write(ptr.add(idx), f());
            });
            self.len = new_len;
        } else {
            self.truncate(new_len);
        }
    }

    /// Retains only the elements specified by the predicate,
    /// passing a mutable reference to it.
    ///
    /// In other words, remove all elements e for which `f(&mut e)`
    /// returns `false`. This method operates in place, visiting each
    /// element exactly once in the original order, and preserves the
    /// order of the retained elements.
    ///
    /// # Time complexity
    /// O(N)
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::ArrayVec;
    /// let mut vec = ArrayVec::from([1, 2, 3, 4]);
    /// vec.retain_mut(|v|{
    ///     *v += 10;
    ///     *v % 2 != 0
    /// });
    /// assert_eq!(vec.len(), 2);
    /// assert_eq!(vec, [11, 13]);
    /// ```
    pub fn retain_mut<F: FnMut(&mut T) -> bool>(&mut self, mut f: F) {
        let mut count = 0usize;
        let base_ptr = self.as_mut_ptr();
        for index in 0..self.len {
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
        self.len = count;
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements e for which `f(&e)` returns `false`.
    /// This method operates in place, visiting each element exactly once in the original order,
    /// and preserves the order of the retained elements.
    ///
    /// # Time complexity
    /// O(N)
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::ArrayVec;
    /// let mut vec = ArrayVec::from([1, 2, 3, 4]);
    ///
    /// vec.retain(|v| *v % 2 == 0);
    ///
    /// assert_eq!(vec.len(), 2);
    /// assert_eq!(vec.pop(), Some(4));
    /// ```
    #[inline]
    pub fn retain<F: FnMut(&T) -> bool>(&mut self, mut f: F) {
        self.retain_mut(|v| f(v));
    }

    /// Removes all but the first of consecutive elements in the vector
    /// satisfying a given equality relation.
    ///
    /// # Time Complexity
    ///
    /// O(N)
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::ArrayVec;
    /// let mut vec = ArrayVec::from(["foo", "bar", "Bar", "baz", "bar"]);
    ///
    /// vec.dedup_by(|a, b| a.eq_ignore_ascii_case(b));
    ///
    /// assert_eq!(vec, ["foo", "bar", "baz", "bar"]);
    /// ```
    pub fn dedup_by<F: FnMut(&mut T, &mut T) -> bool>(&mut self, mut same_bucket: F) {
        if self.len <= 1 {
            return;
        }

        let ptr = self.as_mut_ptr();
        let mut left = 0usize;

        unsafe {
            let mut p_l = ptr.add(left);
            for right in 1..self.len {
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

    /// Removes all but the first of consecutive elements in the vector
    /// that resolve to the same key.
    ///
    ///
    /// # Time Complexity
    ///
    /// O(N)
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::ArrayVec;
    /// let mut vec = ArrayVec::from([10, 20, 21, 30, 20]);
    ///
    /// vec.dedup_by_key(|i| *i / 10);
    ///
    /// assert_eq!(vec, [10, 20, 30, 20]);
    /// ```
    #[inline]
    pub fn dedup_by_key<F, K>(&mut self, mut key: F)
    where
        F: FnMut(&mut T) -> K,
        K: PartialEq,
    {
        self.dedup_by(|a, b| key(a) == key(b));
    }

    /// Returns the remaining spare capacity of the vector as a slice of `MaybeUninit<T>`.
    ///
    /// The returned slice can be used to fill the vector with data (e.g. by reading from a file)
    /// before marking the data as initialized using the [`set_len`](ArrayVec::set_len) method.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::ArrayVec;
    /// // Allocate vector big enough for 10 elements.
    /// let mut v = ArrayVec::<i32, 10>::new();
    ///
    /// // Fill in the first 3 elements.
    /// let uninit = v.spare_capacity_mut();
    /// uninit[0].write(0);
    /// uninit[1].write(1);
    /// uninit[2].write(2);
    ///
    /// // Mark the first 3 elements of the vector as being initialized.
    /// unsafe {
    ///     v.set_len(3);
    /// }
    ///
    /// assert_eq!(v, [0, 1, 2]);
    /// ```
    #[inline]
    pub const fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        unsafe { slice::from_raw_parts_mut(self.data.as_mut_ptr().add(self.len), N - self.len) }
    }
}

// -----------------------------------------------------------------------------
// Common

super::utils::impl_common_traits!(ArrayVec<T, N>);

impl<T, U, const N: usize> PartialEq<ArrayVec<U, N>> for ArrayVec<T, N>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &ArrayVec<U, N>) -> bool {
        PartialEq::eq(self.as_slice(), other.as_slice())
    }
}

impl<T: PartialEq, const N: usize> ArrayVec<T, N> {
    /// Removes consecutive duplicate elements in the vector
    /// according to the [`PartialEq`] trait implementation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::ArrayVec;
    /// let mut vec = ArrayVec::from([1, 2, 2, 3, 2]);
    ///
    /// vec.dedup();
    ///
    /// assert_eq!(vec.as_slice(), [1, 2, 3, 2]);
    /// ```
    #[inline]
    pub fn dedup(&mut self) {
        self.dedup_by(|x, y| PartialEq::eq(x, y));
    }
}

impl<T: Clone, const N: usize> Clone for ArrayVec<T, N> {
    fn clone(&self) -> Self {
        let mut vec = Self::new();
        for item in self.as_slice() {
            unsafe { vec.push_unchecked(item.clone()) };
        }
        vec
    }

    fn clone_from(&mut self, source: &Self) {
        self.clear();
        for item in source.as_slice() {
            unsafe { self.push_unchecked(item.clone()) };
        }
    }
}

impl<'a, T: 'a + Clone, const N: usize> Extend<&'a T> for ArrayVec<T, N> {
    /// Clone values from iterators.
    ///
    /// # Panics
    /// Insufficient capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::ArrayVec;
    /// let mut vec: ArrayVec<i32, 5> = ArrayVec::new();
    ///
    /// vec.extend(&[1, 2, 3]);
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        for item in iter {
            self.push(item.clone());
        }
    }
}

impl<T, const N: usize> Extend<T> for ArrayVec<T, N> {
    /// Extends a collection with the contents of an iterator.
    ///
    /// # Panics
    /// Insufficient capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::ArrayVec;
    /// let mut vec: ArrayVec<i32, 5> = ArrayVec::new();
    ///
    /// vec.extend([1, 2, 3]);
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.push(item);
        }
    }
}

impl<T, const N: usize> FromIterator<T> for ArrayVec<T, N> {
    /// # Panics
    /// Insufficient capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::ArrayVec;
    /// let mut vec = <ArrayVec<i32, 3>>::from_iter([1, 2, 3].into_iter());
    ///
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut vec = Self::new();
        for item in iter {
            vec.push(item);
        }
        vec
    }
}

// -----------------------------------------------------------------------------
// From / Into

impl<T, const N: usize> From<ArrayVec<T, N>> for Vec<T> {
    fn from(v: ArrayVec<T, N>) -> Vec<T> {
        v.into_vec()
    }
}

impl<T, const N: usize> From<ArrayVec<T, N>> for Box<[T]> {
    fn from(v: ArrayVec<T, N>) -> Box<[T]> {
        v.into_vec().into_boxed_slice()
    }
}

impl<T, const N: usize, const P: usize> TryFrom<ArrayVec<T, N>> for [T; P] {
    type Error = ArrayVec<T, N>;

    fn try_from(mut vec: ArrayVec<T, N>) -> Result<[T; P], ArrayVec<T, N>> {
        if vec.len() != P {
            return Err(vec);
        }

        let src = vec.as_ptr();
        unsafe { vec.set_len(0) };
        let array = unsafe { ptr::read(src as *const [T; P]) };
        Ok(array)
    }
}

impl<T, const N: usize> From<[T; N]> for ArrayVec<T, N> {
    fn from(array: [T; N]) -> Self {
        let array = ManuallyDrop::new(array);
        let mut vec = <ArrayVec<T, N>>::new();
        let dst = &raw mut vec.data;
        let src = &raw const *array as *const [MaybeUninit<T>; N];
        unsafe {
            ptr::copy_nonoverlapping(src, dst, 1);
            vec.set_len(N);
        }
        vec
    }
}

// -----------------------------------------------------------------------------
// IntoIter

/// An iterator that consumes a [`ArrayVec`] and yields its items by value.
///
/// # Examples
///
/// ```
/// # use fastvec::ArrayVec;
///
/// let vec = ArrayVec::from(["1", "2", "3"]);
/// let mut iter = vec.into_iter();
///
/// assert_eq!(iter.next(), Some("1"));
///
/// let vec: Vec<&'static str> = iter.collect();
/// assert_eq!(vec, ["2", "3"]);
/// ```
pub struct IntoIter<T, const N: usize> {
    vec: ManuallyDrop<ArrayVec<T, N>>,
    index: usize,
}

unsafe impl<T: Send, const N: usize> Send for IntoIter<T, N> {}
unsafe impl<T: Sync, const N: usize> Sync for IntoIter<T, N> {}

impl<T, const N: usize> IntoIterator for ArrayVec<T, N> {
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
        if self.index < self.vec.len {
            unsafe {
                ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                    self.vec.as_mut_ptr().add(self.index),
                    self.vec.len - self.index,
                ));
            }
        }
    }
}

impl<T, const N: usize> Default for IntoIter<T, N> {
    fn default() -> Self {
        Self {
            vec: ManuallyDrop::new(ArrayVec::new()),
            index: 0,
        }
    }
}

// -----------------------------------------------------------------------------
// Drain

/// An iterator that removes the items from a [`ArrayVec`]
/// and yields them by value.
///
/// See [`ArrayVec::drain`] .
pub struct Drain<'a, T: 'a, const N: usize> {
    tail_start: usize,
    tail_len: usize,
    iter: slice::Iter<'a, T>,
    vec: NonNull<ArrayVec<T, N>>,
}

impl<T, const N: usize> ArrayVec<T, N> {
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
    /// # use fastvec::ArrayVec;
    /// let mut v = ArrayVec::from([1, 2, 3]);
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
/// See [`ArrayVec::extract_if`] .
pub struct ExtractIf<'a, T, F: FnMut(&mut T) -> bool, const N: usize> {
    vec: &'a mut ArrayVec<T, N>,
    idx: usize,
    end: usize,
    del: usize,
    old_len: usize,
    pred: F,
}

impl<T, const N: usize> ArrayVec<T, N> {
    /// Creates an iterator which uses a closure to determine
    /// if an element in the range should be removed.
    ///
    /// See more information in [`Vec::extract_if`].
    ///
    /// # Panics
    /// Panics if the range is out of bounds.
    ///
    /// # Examples
    ///
    /// Splitting a vector into even and odd values, reusing the original vector:
    ///
    /// ```
    /// # use fastvec::ArrayVec;
    /// let mut numbers = ArrayVec::from([1, 2, 3, 4, 5, 6, 8, 9, 11, 13, 14, 15]);
    ///
    /// let evens = numbers.extract_if(.., |x| *x % 2 == 0).collect::<ArrayVec<_, 10>>();
    /// let odds = numbers;
    ///
    /// assert_eq!(evens, [2, 4, 6, 8, 14]);
    /// assert_eq!(odds, [1, 3, 5, 9, 11, 13, 15]);
    /// ```
    ///
    /// Using the range argument to only process a part of the vector:
    ///
    /// ```
    /// # use fastvec::ArrayVec;
    /// let mut items = ArrayVec::from([0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2]);
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
    use super::ArrayVec;
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

        let mut vec = ArrayVec::<Tracker, 8>::new();
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

        // 1 consumed + 2 dropped by Drain::drop in the ZST path.
        assert_eq!(DROPS.load(Ordering::SeqCst), 3);

        drop(vec);
        assert_eq!(DROPS.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn drop_vec() {
        define_tracker!();

        DROPS.store(0, Ordering::SeqCst);
        {
            let mut vec = ArrayVec::<Tracker, 4>::new();
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

        let mut vec = ArrayVec::<Tracker, 4>::new();
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

        let mut vec = ArrayVec::<Tracker, 4>::new();
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

        let mut vec = ArrayVec::<Tracker, 8>::new();
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

        let mut vec = ArrayVec::<Tracker, 8>::new();
        for id in 0..6 {
            vec.push(Tracker { id });
        }

        let removed: ArrayVec<Tracker, 8> = vec.extract_if(.., |t| t.id % 2 == 0).collect();
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);

        drop(removed);
        assert_eq!(DROPS.load(Ordering::SeqCst), 3);

        drop(vec);
        assert_eq!(DROPS.load(Ordering::SeqCst), 6);
    }
}
