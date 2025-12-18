use alloc::{boxed::Box, vec::Vec};
use core::{
    fmt,
    iter::FusedIterator,
    mem::{self, ManuallyDrop, MaybeUninit},
    ptr, slice,
};

use crate::utils::{IsZST, cold_path, zst_init};

/// A vector stored on the stack with a fixed capacity.
///
/// This is useful when the data is small and the maximum quantity is determined.
///
/// It mirrors most of the API of [`Vec`], but maintains the same high efficiency as `[T; N]`.
///
/// # Panics
/// Any operation that causes `len > capacity`.
///
/// # Examples
///
/// ```
/// use fastvec::StackVec;
///
/// // Allocate uninitialized space for 10 elements on the stack
/// let mut vec: StackVec<String, 10> = StackVec::new();
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
/// but the maximum capacity is still valid and cannot overflow.
///
/// We have made many optimizations to zero size types, many functions only
/// modify the length, such as `push`, `copy_from_raw` ...
pub struct StackVec<T, const N: usize> {
    pub(crate) data: [MaybeUninit<T>; N],
    pub(crate) len: usize,
}

unsafe impl<T, const N: usize> Send for StackVec<T, N> where T: Send {}
unsafe impl<T, const N: usize> Sync for StackVec<T, N> where T: Sync {}

impl<T, const N: usize> Drop for StackVec<T, N> {
    // Internal data using `MaybeUninit`, we need to call `drop` manually.
    fn drop(&mut self) {
        if self.len > 0 {
            // SAFETY: Ensure the validity of data within the range.
            unsafe {
                ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.as_mut_ptr(), self.len));
            }
        }
    }
}

/// Creates a [`StackVec`] containing the arguments.
///
/// The syntax is similar to [`vec!`](https://doc.rust-lang.org/std/macro.vec.html).
///
/// You must explicitly specify the container capacity.
/// The number of elements cannot exceed the capacity.
///
/// Non-params macro is equal to [`StackVec::new`].
///
/// # Panics
/// Panics if the number of elements exceeds the capacity.
///
/// # Examples
///
/// ```
/// # use fastvec::{stackvec, StackVec};
/// let vec: StackVec<String, 10> = stackvec![];
/// let vec: StackVec<i64, 10> = stackvec![1; 5]; // Need to support Clone.
/// let vec: StackVec<_, 10> = stackvec![1, 2, 3, 4];
/// ```
#[macro_export]
macro_rules! stackvec {
    [] => { $crate::StackVec::new() };
    [$elem:expr; $n:expr] => { $crate::StackVec::from_elem($elem, $n) };
    [$($item:expr),+ $(,)?] => { $crate::StackVec::from_buf([ $($item),+ ]) };
}

impl<T, const N: usize> StackVec<T, N> {
    /// Constructs a new, empty `StackVec` on the stack with the specified capacity.
    ///
    /// The capacity must be provided at compile time via the const generic parameter.
    ///
    /// Note that the stack memory is allocated when the `StackVec` is instantiated.
    /// The capacity should not be too large to avoid stack overflow.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::StackVec;
    /// let mut vec: StackVec<i32, 8> = StackVec::new();
    /// ```
    #[inline]
    pub const fn new() -> Self {
        Self {
            // SAFETY: Full buffer uninitialized to internal uninitialized is safe.
            data: unsafe { MaybeUninit::<[MaybeUninit<T>; N]>::uninit().assume_init() },
            len: 0,
        }
    }

    /// Modify the capacity of the container.
    ///
    /// If the capacity is insufficient, [`StackVec::truncate`] will be called.
    ///
    /// # Example
    ///
    /// ```
    /// # use fastvec::{StackVec, stackvec};
    /// let vec: StackVec<i32, 5> = stackvec![1, 2, 3, 4];
    ///
    /// let mut vec: StackVec<i32, 10> = vec.force_cast();
    ///
    /// vec.push(5);
    /// vec.push(6);
    /// assert_eq!(vec, [1, 2, 3, 4, 5, 6]);
    /// ```
    #[inline]
    pub fn force_cast<const P: usize>(mut self) -> StackVec<T, P> {
        self.truncate(P);
        let mut vec = <StackVec<T, P>>::new();
        if !T::IS_ZST {
            unsafe {
                ptr::copy_nonoverlapping(self.as_ptr(), vec.as_mut_ptr(), self.len);
            }
        }
        vec.len = self.len;
        self.len = 0;
        vec
    }

    /// Returns a raw pointer to the vector’s buffer, or a dangling pointer
    /// valid for zero-sized reads if `T` is a zero-sized type.
    ///
    /// The caller must ensure that the vector outlives the pointer this function returns, or else it will end up dangling.
    ///
    /// Modifying the vector will **not** cause its buffer to be reallocated.
    #[inline(always)]
    pub const fn as_ptr(&self) -> *const T {
        &raw const self.data as *const T
    }

    /// Returns a raw mutable pointer to the vector’s buffer, or a dangling pointer
    /// valid for zero-sized reads if `T` is a zero-sized type.
    ///
    /// The caller must ensure that the vector outlives the pointer this function returns, or else it will end up dangling.
    ///
    /// Modifying the vector will **not** cause its buffer to be reallocated.
    #[inline(always)]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        &raw mut self.data as *mut T
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

    /// Returns the number of elements in the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::StackVec;
    /// let vec = StackVec::<String, 5>::new();
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
    /// # use fastvec::{StackVec, stackvec};
    /// let mut v = StackVec::<i32, 5>::new();
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
    /// # use fastvec::{StackVec, stackvec};
    /// let mut v = StackVec::<i32, 3>::new();
    /// assert!(!v.is_full());
    ///
    /// v.extend([1, 2, 3]);
    /// assert!(v.is_full());
    /// ```
    #[inline(always)]
    pub const fn is_full(&self) -> bool {
        self.len >= N
    }

    /// Returns the maximum number of elements the vector can hold.
    ///
    /// The capacity is fixed at compile time by the const generic parameter `N`.
    ///
    /// # Examples
    /// ```
    /// # use fastvec::StackVec;
    /// let vec = StackVec::<String, 5>::new();
    /// assert_eq!(vec.capacity(), 5);
    /// ```
    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Copy data from a ptr and create a [`StackVec`].
    ///
    /// This does not check for length overflow, but overflow is an undefined behavior.
    ///
    /// Since the container is stored on the stack, it copies the target value through
    /// [`ptr::copy_nonoverlapping`], and you need to ensure that the target will not be dropped again.
    ///
    /// For zero sized type, only the length will be set (no copy).
    ///
    /// # Safety
    ///
    /// This is highly unsafe, due to the number of invariants that aren’t checked:
    /// - `length` needs to be less than or equal to capacity `N`.
    /// - `T` type needs to be the same size and alignment that it was allocated with.
    /// - It is necessary to avoid the incoming data being dropped twice.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::StackVec;
    /// let mut datas = ["1".to_string(), "2".to_string()];
    ///
    /// let src = datas.as_mut_ptr() as *mut String;
    ///
    /// let vec = unsafe{ StackVec::<String, 5>::copy_from_raw(src, 2) };
    ///
    /// assert_eq!(vec.len(), 2);
    ///
    /// core::mem::forget(datas);
    /// ```
    #[inline(always)]
    pub const unsafe fn copy_from_raw(ptr: *const T, length: usize) -> Self {
        debug_assert!(length <= N);

        let mut vec = Self::new();

        // This judgment can be optimized by compiler.
        if !T::IS_ZST {
            unsafe {
                ptr::copy_nonoverlapping(ptr, vec.as_mut_ptr(), length);
            }
        }

        vec.len = length;
        vec
    }

    /// Creates a [`StackVec`] from an array.
    ///
    /// Copies elements from the provided array into the StackVec.
    ///
    /// # Panics
    /// Panics if the length exceeds the capacity `N`.
    ///
    /// # Examples
    /// ```
    /// # use fastvec::StackVec;
    /// let vec: StackVec<i32, 5> = StackVec::from_buf([1, 2, 3]);
    /// assert_eq!(vec.len(), 3);
    /// ```
    #[inline]
    pub const fn from_buf<const P: usize>(arr: [T; P]) -> Self {
        assert!(P <= N, "length overflow during `from_buf`");

        unsafe { Self::from_buf_unchecked(arr) }
    }

    /// Creates a StackVec from an array without checking bounds.
    ///
    /// # Safety
    /// length <= capacity `N`.
    #[inline(always)]
    pub const unsafe fn from_buf_unchecked<const P: usize>(arr: [T; P]) -> Self {
        unsafe {
            let vec = Self::copy_from_raw(arr.as_ptr(), P);
            core::mem::forget(arr);
            vec
        }
    }

    /// Converts a [`Vec`] into a [`StackVec`].
    ///
    /// This function copies data from the `Vec` into the `StackVec`,
    /// then clears the `Vec`.
    ///
    /// # Panics
    /// Panics if the length exceeds the capacity `N`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::StackVec;
    /// let mut vec = vec![1, 2, 3,  4];
    /// let vec = StackVec::<i32, 5>::from_vec(&mut vec);
    ///
    /// assert_eq!(vec.len(), 4);
    /// assert_eq!(vec.capacity(), 5);
    /// ```
    #[inline]
    pub fn from_vec(vec: &mut Vec<T>) -> Self {
        assert!(vec.len() <= N, "length overflow during `from_vec`");

        unsafe { Self::from_vec_unchecked(vec) }
    }

    /// Converts a [`Vec`] to a [`StackVec`] without checking the length.
    ///
    /// This function copies data from the `Vec` into the `StackVec`,
    /// then clears the `Vec`.
    ///
    /// # Safety
    /// length <= capacity `N`.
    #[inline(always)]
    pub unsafe fn from_vec_unchecked(vec: &mut Vec<T>) -> Self {
        unsafe {
            let res = Self::copy_from_raw(vec.as_ptr(), vec.len());
            vec.set_len(0);
            res
        }
    }

    /// Converts a [`StackVec`] to a [`Vec`].
    ///
    /// Allocates exactly `len` capacity and transfers the data to the heap.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::StackVec;
    /// let mut vec = StackVec::<String, 5>::new();
    /// vec.push("123".to_string());
    ///
    /// let vec = vec.into_vec();
    /// assert_eq!(vec.len(), 1);
    /// assert_eq!(vec.capacity(), 1);
    /// ```
    #[inline]
    pub fn into_vec(&mut self) -> Vec<T> {
        let mut vec: Vec<T> = Vec::with_capacity(self.len);

        unsafe {
            ptr::copy_nonoverlapping(self.as_ptr(), vec.as_mut_ptr(), self.len);
            vec.set_len(self.len);
            self.len = 0;
        }

        vec
    }

    /// Converts a [`StackVec`] into a [`Box<[T]>`](Box).
    #[inline]
    pub fn into_boxed_slice(&mut self) -> Box<[T]> {
        self.into_vec().into_boxed_slice()
    }

    /// Converts a [`StackVec`] to a [`Vec`] with the specified capacity.
    ///
    /// If the specified capacity is less than the length,
    /// the length will be used instead of the specified value.
    #[inline]
    pub fn into_vec_with_capacity(&mut self, capacity: usize) -> Vec<T> {
        let mut vec: Vec<T> = Vec::with_capacity(capacity.max(self.len));

        unsafe {
            ptr::copy_nonoverlapping(self.as_ptr(), vec.as_mut_ptr(), self.len);
            vec.set_len(self.len);
            self.len = 0;
        }

        vec
    }

    /// Converts a [`StackVec`] to a [`Vec`] with the specified capacity.
    ///
    /// # Safety
    /// The caller must ensure `len <= capacity`.
    #[inline(always)]
    pub(crate) unsafe fn into_vec_with_capacity_unchecked(&mut self, capacity: usize) -> Vec<T> {
        let mut vec: Vec<T> = Vec::with_capacity(capacity);

        unsafe {
            core::hint::assert_unchecked(self.len <= capacity);
            ptr::copy_nonoverlapping(self.as_ptr(), vec.as_mut_ptr(), self.len);
            vec.set_len(self.len);
            self.len = 0;
        }

        vec
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
    /// # use fastvec::StackVec;
    /// let mut vec = StackVec::<i32, 5>::new();
    /// vec.push(1);
    /// vec.push(2);
    /// assert_eq!(vec.len(), 2);
    /// ```
    #[inline(always)]
    pub const fn push(&mut self, value: T) {
        let len = self.len;
        assert!(len < N, "length overflow during `push`");

        if T::IS_ZST {
            mem::forget(value);
        } else {
            unsafe {
                ptr::write(self.as_mut_ptr().add(len), value);
            }
        }

        self.len = len + 1;
    }

    /// Appends an element to the back of the vector without bounds checking.
    ///
    /// # Safety
    /// length < capacity `N` (before `push`)
    #[inline(always)]
    pub const unsafe fn push_unchecked(&mut self, value: T) {
        let len: usize = self.len;

        if T::IS_ZST {
            mem::forget(value);
        } else {
            unsafe {
                ptr::write(self.as_mut_ptr().add(len), value);
            }
        }

        self.len = len + 1;
    }

    /// Removes an item from the end of the vector and returns it, or `None` if empty.
    ///
    /// # Time complexity
    /// O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::StackVec;
    /// let mut vec = StackVec::<i32, 5>::new();
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
            cold_path();
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

    /// Removes and returns the last element from a vector if the predicate returns `true`,
    /// or `None` if the predicate returns false or the vector is empty (the predicate will not be called in that case).
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{stackvec, StackVec};
    /// let mut vec: StackVec<_, 5> = stackvec![1, 2, 3, 4];
    /// let pred = |x: &mut i32| *x % 2 == 0;
    /// assert_eq!(vec.pop_if(pred), Some(4));
    /// assert_eq!(vec, [1, 2, 3]);
    /// assert_eq!(vec.pop_if(pred), None);
    /// ```
    #[inline]
    pub fn pop_if(&mut self, predicate: impl FnOnce(&mut T) -> bool) -> Option<T> {
        if self.len == 0 {
            cold_path();
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
    /// Panics if `index > len`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::StackVec;
    /// let mut vec = StackVec::<i32, 5>::new();
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

        unsafe {
            self.insert_unchecked(index, element);
        }
    }

    /// Inserts an element at position `index` within the vector, without bounds checking.
    ///
    /// # Safety
    /// - index <= len (before insertion)
    #[inline(always)]
    pub(crate) const unsafe fn insert_unchecked(&mut self, index: usize, element: T) {
        debug_assert!(index <= self.len, "insertion index should be <= len");
        debug_assert!(self.len < N, "length overflow during `insert`");

        if T::IS_ZST {
            mem::forget(element);
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

    /// Removes an element from the vector and returns it.
    ///
    /// The removed element is replaced by the last element of the vector.
    ///
    /// This does not preserve ordering of the remaining elements, but is O(1).
    /// If you need to preserve the element order, use [`remove`](StackVec::remove) instead.
    ///
    /// # Panics
    ///
    /// Panics if `index >= len`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{StackVec, stackvec};
    /// let mut vec: StackVec<i32, 5> = stackvec![1, 2, 3];
    ///
    /// let x = vec.swap_remove(0); // swap 1 3
    /// assert_eq!(x, 1);
    /// assert_eq!(vec, [3, 2]);
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
    /// # use fastvec::{StackVec, stackvec};
    /// let mut vec: StackVec<i32, 5> = stackvec![1, 2, 3];
    ///
    /// let x = vec.remove(1);
    /// assert_eq!(x, 2);
    /// assert_eq!(vec.len(), 2);
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

    /// Shortens the vector, keeping the first len elements and dropping the rest.
    ///
    /// If len is greater or equal to the vector’s current length, this has no effect.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{StackVec, stackvec};
    /// let mut vec: StackVec::<_, 5> = stackvec![1; 5];
    /// let x = vec.truncate(2);
    /// assert_eq!(vec.len(), 2);
    /// ```
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        if self.len > len {
            unsafe {
                ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                    self.as_mut_ptr().add(len),
                    self.len - len,
                ))
            }
            self.len = len;
        }
    }

    /// Extracts a slice containing the entire vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{StackVec, stackvec};
    /// let mut vec: StackVec::<_, 5> = stackvec![1, 2, 3, 4];
    ///
    /// let slice = vec.as_slice();
    /// assert_eq!(slice, [1, 2, 3, 4]);
    /// ```
    #[inline]
    pub const fn as_slice(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.as_ptr(), self.len) }
    }

    /// Extracts a mutable slice containing the entire vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{StackVec, stackvec};
    /// let mut vec: StackVec::<_, 5> = stackvec![1, 2, 3, 4];
    /// let slice = vec.as_mut_slice();
    ///
    /// slice[3] = 5;
    ///
    /// assert_eq!(vec, [1, 2, 3, 5]);
    /// ```
    #[inline]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len) }
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
    /// # use fastvec::{StackVec, stackvec};
    /// let mut vec: StackVec::<_, 5> = stackvec![1, 2, 3, 4];
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

    /// Retains only the elements specified by the predicate, passing a mutable reference to it.
    ///
    /// In other words, remove all elements e for which `f(&mut e)` returns `false`.
    /// This method operates in place, visiting each element exactly once in the original order,
    /// and preserves the order of the retained elements.
    ///
    /// # Time complexity
    /// O(N)
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{StackVec, stackvec};
    /// let mut vec: StackVec::<_, 5> = stackvec![1, 2, 3, 4];
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

    /// Removes all but the first of consecutive elements in the vector that resolve to the same key.
    ///
    /// See [`Vec::dedup_by_key`].
    ///
    /// # Time Complexity
    ///
    /// O(N)
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{StackVec, stackvec};
    /// let mut vec: StackVec<_, 5> = stackvec![10, 20, 21, 30, 20];
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

    /// Removes all but the first of consecutive elements in the vector satisfying a given equality relation.
    ///
    /// See [`Vec::dedup_by`].
    ///
    /// # Time Complexity
    ///
    /// O(N)
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{StackVec, stackvec};
    /// let mut vec: StackVec<_, 5> = stackvec!["foo", "bar", "Bar", "baz", "bar"];
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
                        core::mem::swap(&mut *p_r, &mut *p_l);
                    }
                }
            }
        }
        self.truncate(left + 1);
    }

    /// Moves all the elements of other into self, leaving other empty.
    ///
    /// # Panics
    ///
    /// Panics if the new length exceeds `N`.
    ///
    /// # Examples
    /// ```
    /// # use fastvec::{StackVec, stackvec};
    /// let mut vec1: StackVec<_, 6> = stackvec![1, 2, 3, 4];
    /// let mut vec2: StackVec<_, 4> = stackvec![5, 6];
    /// vec1.append(&mut vec2);
    /// assert_eq!(vec1, [1, 2, 3, 4, 5, 6]);
    /// assert_eq!(vec2, []);
    /// ```
    #[inline]
    pub fn append<const P: usize>(&mut self, other: &mut StackVec<T, P>) {
        let other_len = other.len();
        let self_len = self.len;
        assert!(self_len + other_len <= N, "length overflow during `append`");

        if !T::IS_ZST {
            unsafe {
                ptr::copy_nonoverlapping(
                    other.as_ptr(),
                    self.as_mut_ptr().add(self_len),
                    other_len,
                );
            }
        }

        self.len = self_len + other_len;
        other.len = 0;
    }

    /// Moves all the elements of [`Vec`] into `self`, leaving the source empty.
    ///
    /// Because the type and quantity are known, this will be more efficient than [`Extend`].
    ///
    /// # Panics
    ///
    /// Panics if the new length exceeds `N`.
    ///
    /// # Examples
    /// ```
    /// # use fastvec::{StackVec, stackvec};
    /// let mut vec1: StackVec<_, 6> = stackvec![1, 2, 3, 4];
    /// let mut vec2: Vec<_> = vec![5, 6];
    /// vec1.append_vec(&mut vec2);
    /// assert_eq!(vec1, [1, 2, 3, 4, 5, 6]);
    /// assert_eq!(vec2, []);
    /// ```
    #[inline]
    pub fn append_vec(&mut self, other: &mut Vec<T>) {
        let other_len = other.len();
        let self_len = self.len;
        assert!(self_len + other_len <= N, "length overflow during `append`");

        if !T::IS_ZST {
            unsafe {
                ptr::copy_nonoverlapping(
                    other.as_ptr(),
                    self.as_mut_ptr().add(self_len),
                    other_len,
                );
            }
        }

        unsafe {
            self.len = self_len + other_len;
            other.set_len(0);
        }
    }

    /// Moves all the elements of `self` into the given [`Vec`], leaving `self` empty.
    ///
    /// Because the type and quantity are known, this will be more efficient than [`Extend`].
    ///
    /// # Examples
    /// ```
    /// # use fastvec::{StackVec, stackvec};
    /// let mut vec1: Vec<_> = vec![1, 2, 3, 4];
    /// let mut vec2: StackVec<_, 4> = stackvec![5, 6];
    /// vec2.append_to_vec(&mut vec1);
    /// assert_eq!(vec1, [1, 2, 3, 4, 5, 6]);
    /// assert_eq!(vec2, []);
    /// ```
    #[inline]
    pub fn append_to_vec(&mut self, other: &mut Vec<T>) {
        let other_len = other.len();
        let self_len = self.len;

        other.reserve(self_len);

        if !T::IS_ZST {
            unsafe {
                ptr::copy_nonoverlapping(
                    self.as_ptr(),
                    other.as_mut_ptr().add(other_len),
                    self_len,
                );
            }
        }

        unsafe {
            other.set_len(other_len + self_len);
            self.len = 0;
        }
    }

    /// Clears the vector, removing all values.
    ///
    /// # Examples
    /// ```
    /// # use fastvec::{StackVec, stackvec};
    /// let mut v: StackVec::<i32, 5> = stackvec![1, 2, 3];
    ///
    /// v.clear();
    ///
    /// assert!(v.is_empty());
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        if self.len > 0 {
            unsafe {
                ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.as_mut_ptr(), self.len))
            }
            self.len = 0;
        }
    }

    /// Splits the collection into two at the given index.
    ///
    /// # Panics
    /// Panics if `at > len`.
    ///
    /// # Examples
    /// ```
    /// # use fastvec::{StackVec, stackvec};
    /// let mut vec: StackVec::<_, 5> = stackvec!['a', 'b', 'c'];
    /// let vec2 = vec.split_off(1);
    ///
    /// assert_eq!(vec, ['a']);
    /// assert_eq!(vec2, ['b', 'c']);
    /// ```
    #[inline]
    pub const fn split_off(&mut self, at: usize) -> Self {
        assert!(at <= self.len, "the `at` of split off should be <= len");
        let mut other = Self::new();

        unsafe {
            other.len = self.len - at;
            self.len = at;

            if !T::IS_ZST {
                ptr::copy_nonoverlapping(self.as_ptr().add(at), other.as_mut_ptr(), other.len);
            }
        }
        other
    }

    /// Resizes the [`StackVec`] in-place so that len is equal to new_len.
    ///
    /// # Panics
    /// Panics if the new length exceeds N.
    ///
    /// # Example
    /// ```
    /// # use fastvec::{StackVec, stackvec};
    /// let mut vec: StackVec::<_, 5> = stackvec![1, 2, 3, 4];
    /// vec.resize_with(2, Default::default);
    /// assert_eq!(vec, [1, 2]);
    ///
    /// let mut p = 1;
    /// vec.resize_with(5, || { p *= 2; p });
    /// assert_eq!(vec, [1, 2, 2, 4, 8]);
    /// ```
    pub fn resize_with<F: FnMut() -> T>(&mut self, new_len: usize, mut f: F) {
        assert!(new_len <= N, "length overflow during `resize_with`");

        if new_len < self.len {
            self.truncate(new_len);
        } else {
            for index in self.len..new_len {
                unsafe {
                    ptr::write(self.as_mut_ptr().add(index), f());
                }
            }
            self.len = new_len;
        }
    }

    /// Returns the remaining spare capacity of the vector as a slice of `MaybeUninit<T>`.
    ///
    /// The returned slice can be used to fill the vector with data (e.g. by reading from a file)
    /// before marking the data as initialized using the [`set_len`](StackVec::set_len) method.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{StackVec, stackvec};
    /// // Allocate vector big enough for 10 elements.
    /// let mut v = StackVec::<i32, 10>::new();
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
    #[inline(always)]
    pub const fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        unsafe {
            slice::from_raw_parts_mut(
                { &raw mut self.data as *mut MaybeUninit<T> }.add(self.len),
                N - self.len,
            )
        }
    }
}

impl<T: Clone, const N: usize> StackVec<T, N> {
    /// Creates a [`StackVec`] with `num` copies of `elem`.
    ///
    /// This function requires `T` to implement `Clone`.
    ///
    /// # Panics
    /// Panics if `num > N`.
    ///
    /// # Examples
    /// ```
    /// # use fastvec::StackVec;
    /// let vec: StackVec<i32, 5> = StackVec::from_elem(1, 4);
    /// assert_eq!(vec, [1, 1, 1, 1]);
    /// ```
    #[inline]
    pub fn from_elem(elem: T, num: usize) -> Self {
        assert!(num <= N, "length overflow during `from_elem`");

        let mut vec = Self::new();

        if num != 0 {
            let base_ptr = vec.as_mut_ptr();
            unsafe {
                for index in 1..num {
                    ptr::write(base_ptr.add(index), elem.clone());
                }
                // Reduce one copy.
                ptr::write(base_ptr, elem);
            }
        }

        vec.len = num;
        vec
    }

    /// Resizes the [`StackVec`] in-place so that len is equal to `new_len`.
    ///
    /// # Panics
    ///
    /// Panics if `new_len` > capacity `N`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{StackVec, stackvec};
    /// let mut vec: StackVec<_, 5> = stackvec!["hello"];
    /// vec.resize(3, "world");
    /// assert_eq!(vec, ["hello", "world", "world"]);
    ///
    /// let mut vec: StackVec<_, 5> = stackvec!['a', 'b', 'c', 'd'];
    /// vec.resize(2, '_');
    /// assert_eq!(vec, ['a', 'b']);
    /// ```
    pub fn resize(&mut self, new_len: usize, value: T) {
        assert!(new_len <= N, "length overflow during `resize`");

        if new_len < self.len {
            self.truncate(new_len);
        } else if new_len > self.len {
            unsafe {
                for index in self.len + 1..new_len {
                    ptr::write(self.as_mut_ptr().add(index), value.clone());
                }
                ptr::write(self.as_mut_ptr().add(self.len), value);
            }
            self.len = new_len;
        }
    }

    /// Extends the vector by cloning all elements from the given slice.
    ///
    /// # Panics
    ///
    /// Panics if the length exceeds `N`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{StackVec, stackvec};
    /// let mut vec: StackVec<_, 5> = stackvec![1];
    /// vec.extend_from_slice(&[2, 3, 4]);
    /// assert_eq!(vec.as_slice(), [1, 2, 3, 4]);
    /// ```
    pub fn extend_from_slice(&mut self, other: &[T]) {
        assert!(
            self.len + other.len() <= N,
            "the length should be <= capacity"
        );

        unsafe {
            for item in other {
                ptr::write(self.as_mut_ptr().add(self.len), item.clone());
                self.len += 1;
            }
        }
    }

    /// Clones elements from the given range within the vector and appends them to the end.
    ///
    /// The range `src` must form a valid subslice of the StackVec.
    ///
    /// # Panics
    /// - Starting index is greater than the end index.
    /// - The index is greater than the length of the vector.
    /// - The total length is greater then the capacity `N` .
    ///
    /// # Example
    /// ```
    /// # use fastvec::{StackVec, stackvec};
    /// let mut vec: StackVec<_, 10> = stackvec!['a', 'b', 'c', 'd', 'e'];
    /// vec.extend_from_within(2..);
    /// assert_eq!(vec.as_slice(), ['a', 'b', 'c', 'd', 'e', 'c', 'd', 'e']);
    /// ```
    pub fn extend_from_within<R: core::ops::RangeBounds<usize>>(&mut self, src: R) {
        let (start, end) = crate::utils::split_range_bound(&src, self.len);
        assert!(
            end - start + self.len <= N,
            "the length should be <= capacity"
        );

        unsafe {
            let base_ptr = self.as_mut_ptr();
            for index in start..end {
                ptr::write(base_ptr.add(self.len), (&*base_ptr.add(index)).clone());
                self.len += 1;
            }
        }
    }
}

impl<T: PartialEq, const N: usize> StackVec<T, N> {
    /// Removes consecutive duplicate elements in the vector according to the [`PartialEq`] trait implementation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{StackVec, stackvec};
    /// let mut vec: StackVec<_, 10> = stackvec![1, 2, 2, 3, 2];
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

impl<T, const N: usize, const P: usize> StackVec<[T; P], N> {
    /// Takes a `StackVec<[T; P], N>` and flattens it into a `StackVec<T, S>`.
    ///
    /// # Panics
    /// Panics if `S < P * len`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{StackVec, stackvec};
    /// let mut vec: StackVec<_, 3> = stackvec![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    /// assert_eq!(vec.pop(), Some([7, 8, 9]));
    ///
    /// let mut flattened = vec.into_flattened::<6>();
    /// assert_eq!(flattened, [1, 2, 3, 4, 5, 6]);
    /// ```
    #[inline]
    pub fn into_flattened<const S: usize>(mut self) -> StackVec<T, S> {
        assert!(S >= P * self.len, "the length should be <= capacity");

        let mut vec = StackVec::<T, S>::new();

        unsafe {
            ptr::copy_nonoverlapping(self.as_ptr() as *const T, vec.as_mut_ptr(), self.len * P);
            vec.len = self.len * P;
            self.len = 0;
        }

        vec
    }
}

impl<T, const N: usize> Default for StackVec<T, N> {
    /// Constructs a new, empty `StackVec` on the stack with the specified capacity.
    ///
    /// It's eq to [`StackVec::new`] .
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone, const N: usize> Clone for StackVec<T, N> {
    /// See [`Clone::clone`]
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{StackVec, stackvec};
    /// let vec: StackVec<i32, 5> = stackvec![1, 2 , 3];
    ///
    /// let vec2 = vec.clone();
    /// assert_eq!(vec, [1, 2 , 3]);
    /// assert_eq!(vec, vec2);
    /// ```
    fn clone(&self) -> Self {
        let mut vec = Self::new();
        for item in self.as_slice() {
            unsafe { vec.push_unchecked(item.clone()) };
        }
        vec
    }

    /// See [`Clone::clone_from`]
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{StackVec, stackvec};
    /// let vec: StackVec<i32, 5> = stackvec![1, 2 , 3];
    ///
    /// let mut vec2 = stackvec![];
    /// vec2.clone_from(&vec);
    /// assert_eq!(vec, [1, 2 , 3]);
    /// assert_eq!(vec, vec2);
    /// ```
    fn clone_from(&mut self, source: &Self) {
        self.clear();
        for item in source.as_slice() {
            unsafe { self.push_unchecked(item.clone()) };
        }
    }
}

impl<'a, T: 'a + Clone, const N: usize> Extend<&'a T> for StackVec<T, N> {
    /// Clone values from iterators.
    ///
    /// # Panics
    /// Insufficient capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{StackVec, stackvec};
    /// let mut vec: StackVec<i32, 5> = stackvec![];
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

impl<T, const N: usize> Extend<T> for StackVec<T, N> {
    /// Extends a collection with the contents of an iterator.
    ///
    /// # Panics
    /// Insufficient capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{StackVec, stackvec};
    /// let mut vec: StackVec<i32, 5> = stackvec![];
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

crate::utils::impl_commen_traits!(StackVec<T, N>);

impl<T, U, const N: usize> PartialEq<StackVec<U, N>> for StackVec<T, N>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &StackVec<U, N>) -> bool {
        PartialEq::eq(self.as_slice(), other.as_slice())
    }
}

impl<'a, T: Clone, const N: usize> From<&'a StackVec<T, N>> for alloc::borrow::Cow<'a, [T]> {
    fn from(v: &'a StackVec<T, N>) -> alloc::borrow::Cow<'a, [T]> {
        alloc::borrow::Cow::Borrowed(v.as_slice())
    }
}

impl<'a, T: Clone, const N: usize> From<StackVec<T, N>> for alloc::borrow::Cow<'a, [T]> {
    fn from(mut v: StackVec<T, N>) -> alloc::borrow::Cow<'a, [T]> {
        alloc::borrow::Cow::Owned(v.into_vec())
    }
}

impl<T: Clone, const N: usize> From<&[T]> for StackVec<T, N> {
    /// # Panics
    /// Insufficient capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::StackVec;
    /// let mut vec = <StackVec<i32, 3>>::from([1, 2, 3].as_slice());
    ///
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    fn from(value: &[T]) -> Self {
        assert!(value.len() <= N, "length overflow when `from`");
        let mut vec = Self::new();
        for items in value {
            unsafe {
                vec.push_unchecked(items.clone());
            }
        }
        vec
    }
}

impl<T: Clone, const N: usize, const P: usize> From<&[T; P]> for StackVec<T, N> {
    /// # Panics
    /// Insufficient capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::StackVec;
    /// let mut vec = <StackVec<i32, 3>>::from(&[1, 2, 3]);
    ///
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    fn from(value: &[T; P]) -> Self {
        assert!(P <= N, "length overflow when `from`");
        let mut vec = Self::new();
        for items in value {
            unsafe {
                vec.push_unchecked(items.clone());
            }
        }
        vec
    }
}

impl<T: Clone, const N: usize> From<&mut [T]> for StackVec<T, N> {
    #[inline]
    fn from(value: &mut [T]) -> Self {
        <Self as From<&[T]>>::from(value)
    }
}

impl<T: Clone, const N: usize, const P: usize> From<&mut [T; P]> for StackVec<T, N> {
    #[inline]
    fn from(value: &mut [T; P]) -> Self {
        <Self as From<&[T; P]>>::from(value)
    }
}

impl<T, const N: usize, const P: usize> From<[T; P]> for StackVec<T, N> {
    /// # Panics
    /// Insufficient capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::StackVec;
    /// let mut vec = <StackVec<i32, 3>>::from([1, 2, 3]);
    ///
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    fn from(value: [T; P]) -> Self {
        assert!(P <= N, "length overflow when `from`");
        let mut vec = Self::new();
        unsafe {
            ptr::copy_nonoverlapping(value.as_ptr(), vec.as_mut_ptr(), P);
            vec.len = P;
            mem::forget(value);
        }
        vec
    }
}

impl<T, const N: usize> From<Box<[T]>> for StackVec<T, N> {
    /// # Panics
    /// Insufficient capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::StackVec;
    /// let mut vec = <StackVec<i32, 3>>::from(vec![1, 2, 3].into_boxed_slice());
    ///
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    #[inline]
    fn from(value: Box<[T]>) -> Self {
        Self::from(value.into_vec())
    }
}

impl<T, const N: usize> From<Vec<T>> for StackVec<T, N> {
    /// # Panics
    /// Insufficient capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::StackVec;
    /// let mut vec = <StackVec<i32, 3>>::from(vec![1, 2, 3]);
    ///
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    fn from(mut value: Vec<T>) -> Self {
        assert!(value.len() <= N, "length overflow when `from`");
        unsafe { Self::from_vec_unchecked(&mut value) }
    }
}

impl<T, const N: usize, const P: usize> From<crate::FastVec<T, P>> for StackVec<T, N> {
    /// # Panics
    /// Insufficient capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{StackVec, FastVec};
    /// let mut vec = <StackVec<i32, 3>>::from(
    ///     FastVec::<i32>::from([1, 2, 3])
    /// );
    ///
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    fn from(mut value: crate::FastVec<T, P>) -> Self {
        let len = value.len();
        assert!(len <= N, "length overflow when `from`");
        let vec = value.get();
        vec.len = 0;
        unsafe { Self::copy_from_raw(vec.as_ptr(), len) }
    }
}

impl<T, const N: usize, const P: usize> From<crate::AutoVec<T, P>> for StackVec<T, N> {
    /// # Panics
    /// Insufficient capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{StackVec, AutoVec};
    /// let mut vec = <StackVec<i32, 3>>::from(
    ///     AutoVec::<i32, 3>::from([1, 2, 3])
    /// );
    ///
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    fn from(mut value: crate::AutoVec<T, P>) -> Self {
        let len = value.len();
        assert!(len <= N, "length overflow when `from`");
        unsafe {
            value.set_len(0);
            Self::copy_from_raw(value.as_ptr(), len)
        }
    }
}

impl<T, const N: usize> FromIterator<T> for StackVec<T, N> {
    /// # Panics
    /// Insufficient capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::StackVec;
    /// let mut vec = <StackVec<i32, 3>>::from_iter([1, 2, 3].into_iter());
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

/// An iterator that consumes a [`StackVec`] and yields its items by value.
///
/// # Examples
///
/// ```
/// # use fastvec::{StackVec, stackvec};
///
/// let vec: StackVec<&'static str, 3> = stackvec!["1", "2", "3"];
/// let mut iter = vec.into_iter();
///
/// assert_eq!(iter.next(), Some("1"));
///
/// let vec: Vec<&'static str> = iter.collect();
/// assert_eq!(vec, ["2", "3"]);
/// ```
#[derive(Clone)]
pub struct IntoIter<T, const N: usize> {
    vec: ManuallyDrop<StackVec<T, N>>,
    index: usize,
}

unsafe impl<T, const N: usize> Send for IntoIter<T, N> where T: Send {}
unsafe impl<T, const N: usize> Sync for IntoIter<T, N> where T: Sync {}

impl<T, const N: usize> IntoIterator for StackVec<T, N> {
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
        if self.index < self.vec.len {
            self.index += 1;
            if T::IS_ZST {
                unsafe { Some(zst_init()) }
            } else {
                unsafe { Some(ptr::read(self.vec.as_ptr().add(self.index - 1))) }
            }
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let v = self.vec.len - self.index;
        (v, Some(v))
    }
}

impl<T, const N: usize> DoubleEndedIterator for IntoIter<T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index < self.vec.len {
            self.vec.len -= 1;
            if T::IS_ZST {
                unsafe { Some(zst_init()) }
            } else {
                unsafe { Some(ptr::read(self.vec.as_ptr().add(self.vec.len))) }
            }
        } else {
            None
        }
    }
}

impl<T, const N: usize> ExactSizeIterator for IntoIter<T, N> {
    #[inline]
    fn len(&self) -> usize {
        self.vec.len - self.index
    }
}

impl<T, const N: usize> FusedIterator for IntoIter<T, N> {}

impl<T, const N: usize> Drop for IntoIter<T, N> {
    fn drop(&mut self) {
        if self.index < self.vec.len {
            unsafe {
                ptr::drop_in_place(core::slice::from_raw_parts_mut(
                    self.vec.as_mut_ptr().add(self.index),
                    self.vec.len - self.index,
                ));
            }
        }
    }
}

impl<T, const N: usize> IntoIter<T, N> {
    pub fn as_slice(&self) -> &[T] {
        let len = self.vec.len - self.index;
        unsafe { core::slice::from_raw_parts(self.vec.as_ptr().add(self.index), len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let len = self.vec.len - self.index;
        unsafe { core::slice::from_raw_parts_mut(self.vec.as_mut_ptr().add(self.index), len) }
    }
}

impl<T, const N: usize> Default for IntoIter<T, N> {
    fn default() -> Self {
        Self {
            vec: ManuallyDrop::new(StackVec::new()),
            index: 0,
        }
    }
}

impl<T: fmt::Debug, const N: usize> fmt::Debug for IntoIter<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("IntoIter").field(&self.as_slice()).finish()
    }
}

/// An iterator that removes the items from a [`StackVec`] and yields them by value.
///
/// See [`StackVec::drain`] .
pub struct Drain<'a, T: 'a, const N: usize> {
    tail_start: usize,
    tail_len: usize,
    iter: core::slice::Iter<'a, T>,
    vec: ptr::NonNull<StackVec<T, N>>,
}

impl<T, const N: usize> StackVec<T, N> {
    /// Removes the subslice indicated by the given range from the vector,
    /// returning a double-ended iterator over the removed subslice.
    ///
    /// If the iterator is dropped before being fully consumed, it drops the remaining removed elements.
    ///
    /// The returned iterator keeps a mutable borrow on the vector to optimize its implementation.
    ///
    /// # Panics
    /// Panics if the range has `start_bound > end_bound`, or
    /// if the range is bounded on either end and past the length of the vector.
    ///
    /// See more information in [`Vec::drain`].
    ///
    /// # Examples
    /// ```
    /// # use fastvec::{StackVec, stackvec};
    /// let mut v: StackVec<_, 5> = stackvec![1, 2, 3];
    /// let u: Vec<_> = v.drain(1..).collect();
    /// assert_eq!(v.as_slice(), [1]);
    /// assert_eq!(u, [2, 3]);
    ///
    /// // A full range clears the vector, like `clear()` does
    /// v.drain(..);
    /// assert_eq!(v, []);
    /// ```
    pub fn drain<R: core::ops::RangeBounds<usize>>(&mut self, range: R) -> Drain<'_, T, N> {
        let len = self.len;

        let (start, end) = crate::utils::split_range_bound(&range, len);

        unsafe {
            self.len = start;

            let range_slice = core::slice::from_raw_parts(self.as_ptr().add(start), end - start);

            Drain {
                tail_start: end,
                tail_len: len - end,
                iter: range_slice.iter(),
                vec: core::ptr::NonNull::new_unchecked(self as *mut _),
            }
        }
    }
}

impl<T, const N: usize> Drain<'_, T, N> {
    pub fn as_slice(&self) -> &[T] {
        self.iter.as_slice()
    }
}

impl<T, const N: usize> AsRef<[T]> for Drain<'_, T, N> {
    fn as_ref(&self) -> &[T] {
        self.iter.as_slice()
    }
}

impl<T: fmt::Debug, const N: usize> fmt::Debug for Drain<'_, T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

impl<T, const N: usize> core::iter::FusedIterator for Drain<'_, T, N> {}

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
                        let start = source_vec.len;
                        let tail = self.0.tail_start;
                        if tail != start {
                            let src = source_vec.as_ptr().add(tail);
                            let dst = source_vec.as_mut_ptr().add(start);
                            ptr::copy(src, dst, self.0.tail_len);
                        }
                        source_vec.len = start + self.0.tail_len;
                    }
                }
            }
        }

        let iter = core::mem::take(&mut self.iter);
        let drop_len = iter.len();

        let mut vec = self.vec;

        if core::mem::size_of::<T>() == 0 {
            // ZSTs have no identity, so we don't need to move them around, we only need to drop the correct amount.
            // this can be achieved by manipulating the Vec length instead of moving values out from `iter`.
            unsafe {
                let vec = vec.as_mut();
                let old_len = vec.len();
                vec.len = old_len + drop_len + self.tail_len;
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

/// A splicing iterator for [`StackVec`].
///
/// See [`StackVec::splice`] .
#[derive(Debug)]
pub struct Splice<'a, I: ExactSizeIterator + 'a, const N: usize> {
    pub(super) drain: Drain<'a, I::Item, N>,
    pub(super) replace_with: I,
}

impl<T, const N: usize> StackVec<T, N> {
    /// Creates a splicing iterator that replaces the specified range in the vector
    /// with the given `replace_with` iterator and yields the removed items.
    /// `replace_with` does not need to be the same length as `range`.
    ///
    /// See [`alloc::vec::Splice`] for details; unlike `Vec::splice`, this requires
    /// `replace_with` to implement [`ExactSizeIterator`].
    ///
    /// This is optimal if:
    ///
    /// * The tail (elements in the vector after `range`) is empty,
    /// * or `replace_with` yields elements equal to `range`'s length.
    ///
    /// # Panics
    ///
    /// - if the range has `start_bound > end_bound`.
    /// - if the range is bounded on either end and past the length of the vector.
    /// - result length > capacity `N`
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{stackvec, StackVec};
    /// let mut v: StackVec<_, 5> = stackvec![1, 2, 3, 4];
    /// let new = [7, 8, 9];
    /// let u: Vec<_> = v.splice(1..3, new).collect();
    /// assert_eq!(v, [1, 7, 8, 9, 4]);
    /// assert_eq!(u, [2, 3]);
    /// ```
    ///
    /// Using `splice` to insert new items into a vector efficiently at a specific position
    /// indicated by an empty range:
    ///
    /// ```
    /// # use fastvec::{stackvec, StackVec};
    /// let mut v: StackVec<_, 5> = stackvec![1, 5];
    /// let new = [2, 3, 4];
    /// v.splice(1..1, new);
    /// assert_eq!(v, [1, 2, 3, 4, 5]);
    /// ```
    pub fn splice<R, I>(&mut self, range: R, replace_with: I) -> Splice<'_, I::IntoIter, N>
    where
        R: core::ops::RangeBounds<usize>,
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        Splice {
            drain: self.drain(range),
            replace_with: replace_with.into_iter(),
        }
    }
}

impl<I: ExactSizeIterator, const N: usize> Iterator for Splice<'_, I, N> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.drain.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.drain.size_hint()
    }
}

impl<I: ExactSizeIterator, const N: usize> DoubleEndedIterator for Splice<'_, I, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.drain.next_back()
    }
}

impl<I: ExactSizeIterator, const N: usize> ExactSizeIterator for Splice<'_, I, N> {
    fn len(&self) -> usize {
        self.drain.len()
    }
}

impl<'a, I: ExactSizeIterator, const N: usize> Drop for Splice<'a, I, N> {
    fn drop(&mut self) {
        self.drain.by_ref().for_each(drop);
        // At this point draining is done and the only remaining tasks are splicing
        // and moving things into the final place.
        // Which means we can replace the slice::Iter with pointers that won't point to deallocated
        // memory, so that Drain::drop is still allowed to call iter.len(), otherwise it would break
        // the ptr.offset_from_unsigned contract.
        self.drain.iter = (&[]).iter();

        unsafe {
            if self.drain.tail_len == 0 {
                self.drain.vec.as_mut().extend(self.replace_with.by_ref());
                return;
            }

            // There may be more elements. Use the lower bound as an estimate.
            // FIXME: Is the upper bound a better guess? Or something else?
            let exact_len = self.replace_with.len();
            let vec = self.drain.vec.as_mut();

            // Move tail
            let new_tail_start = vec.len + exact_len;
            if new_tail_start != self.drain.tail_start {
                assert!(
                    new_tail_start + self.drain.tail_len <= N,
                    "the result length in splice should be <= capacity"
                );

                let src = vec.as_ptr().add(self.drain.tail_start);
                let dst = vec.as_mut_ptr().add(new_tail_start);
                ptr::copy(src, dst, self.drain.tail_len);

                self.drain.tail_start = new_tail_start;
            }

            let range_slice =
                core::slice::from_raw_parts_mut(vec.as_mut_ptr().add(vec.len), exact_len);

            for place in range_slice {
                let new_item = self
                    .replace_with
                    .next()
                    .expect("ExactSizeIterator::len must be right.");
                ptr::write(place, new_item);
            }
            vec.len += exact_len;
        }
    }
}

/// An iterator which uses a closure to determine if an element should be removed.
///
/// See [`StackVec::extract_if`] .
pub struct ExtractIf<'a, T, F: FnMut(&mut T) -> bool, const N: usize> {
    vec: &'a mut StackVec<T, N>,
    idx: usize,
    end: usize,
    del: usize,
    old_len: usize,
    pred: F,
}

impl<T, const N: usize> StackVec<T, N> {
    /// Creates an iterator which uses a closure to determine if an element in the range should be removed.
    ///
    /// See more information in [`Vec::extract_if`].
    ///
    /// # Panics
    ///
    /// If `range` is out of bounds.
    ///
    /// # Examples
    ///
    /// Splitting a vector into even and odd values, reusing the original vector:
    ///
    /// ```
    /// # use fastvec::{stackvec, StackVec};
    /// let mut numbers: StackVec<_, 20> = stackvec![1, 2, 3, 4, 5, 6, 8, 9, 11, 13, 14, 15];
    ///
    /// let evens = numbers.extract_if(.., |x| *x % 2 == 0).collect::<StackVec<_, 10>>();
    /// let odds = numbers;
    ///
    /// assert_eq!(evens, [2, 4, 6, 8, 14]);
    /// assert_eq!(odds, [1, 3, 5, 9, 11, 13, 15]);
    /// ```
    ///
    /// Using the range argument to only process a part of the vector:
    ///
    /// ```
    /// # use fastvec::{stackvec, StackVec};
    /// let mut items: StackVec<_, 15> = stackvec![0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2];
    /// let ones = items.extract_if(7.., |x| *x == 1).collect::<Vec<_>>();
    /// assert_eq!(items, [0, 0, 0, 0, 0, 0, 0, 2, 2, 2]);
    /// assert_eq!(ones.len(), 3);
    /// ```
    pub fn extract_if<F, R>(&mut self, range: R, filter: F) -> ExtractIf<'_, T, F, N>
    where
        F: FnMut(&mut T) -> bool,
        R: core::ops::RangeBounds<usize>,
    {
        let old_len = self.len;
        let (start, end) = crate::utils::split_range_bound(&range, old_len);

        // Guard against the vec getting leaked (leak amplification)
        self.len = 0;

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
        if self.del > 0 {
            // SAFETY: Trailing unchecked items must be valid since we never touch them.
            unsafe {
                ptr::copy(
                    self.vec.as_ptr().add(self.idx),
                    self.vec.as_mut_ptr().add(self.idx - self.del),
                    self.old_len - self.idx,
                );
            }
        }
        // SAFETY: After filling holes, all items are in contiguous memory.
        self.vec.len = self.old_len - self.del;
    }
}

impl<T: fmt::Debug, F: FnMut(&mut T) -> bool, const N: usize> fmt::Debug
    for ExtractIf<'_, T, F, N>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
