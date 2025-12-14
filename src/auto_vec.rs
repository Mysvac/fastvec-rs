use alloc::{boxed::Box, vec::Vec};
use core::{fmt, iter::FusedIterator, ptr};

use crate::StackVec;

#[derive(Clone, Debug)]
pub enum InnerVec<T, const N: usize> {
    Stack(StackVec<T, N>),
    Heap(Vec<T>),
}

#[derive(Clone, Copy, Debug)]
pub enum InnerVecRef<'a, T, const N: usize> {
    Stack(&'a StackVec<T, N>),
    Heap(&'a Vec<T>),
}

#[derive(Debug)]
pub enum InnerVecMut<'a, T, const N: usize> {
    Stack(&'a mut StackVec<T, N>),
    Heap(&'a mut Vec<T>),
}

/// A vector stored on stack by default, auto move to heap when capacity is insufficient.
///
/// This type is useful when you are unsure of the length of the data but know that the amount of data is small.
/// At this point, using this container can avoid the overhead of heap requests.
///
/// Most methods are similar to [alloc::vec::Vec] .
/// Internally, it is actually an enumeration that stores either [`StackVec`] or [`Vec`].
///
/// # Example
///
/// ```
/// use fastvec::AutoVec;
/// use core::iter::Extend;
///
/// // Allocate uninit memory for 5 elements on the stack,
/// // It can be completed during compilation.
/// let mut vec: AutoVec<&'static str, 5> = AutoVec::new();
///
/// assert_eq!(vec.len(), 0);
/// assert_eq!(vec.capacity(), 5);
///
/// // Then you can use it like alloc::vec::Vec,
/// // The only difference is that data will exist in the stack
/// // if the capacity is sufficient.
/// vec.push("Hello");
/// vec.push("world");
///
/// assert_eq!(vec, ["Hello", "world"]);
///
/// // If the capacity is insufficient,
/// // the data will be automatically moved to the heap.
/// vec.extend(&["2025", "12", "14", "1:15"]);
/// assert!(!vec.in_stack());
/// assert_eq!(vec, ["Hello", "world", "2025", "12", "14", "1:15"]);
///
/// // You can force it to move to the stack,
/// // which will deconstruct the excess content.
/// vec.force_to_stack();
/// assert!(vec.in_stack());
/// assert_eq!(vec, ["Hello", "world", "2025", "12", "14"]);
///
/// // If necessary, convert to Vec.
/// let vec: Vec<&'static str> = vec.into_vec();
/// ```
#[derive(Clone)]
#[repr(transparent)]
pub struct AutoVec<T, const N: usize>(InnerVec<T, N>);

impl<T, const N: usize> From<Vec<T>> for AutoVec<T, N> {
    #[inline]
    fn from(value: Vec<T>) -> Self {
        Self(InnerVec::Heap(value))
    }
}

impl<T, const N: usize> From<StackVec<T, N>> for AutoVec<T, N> {
    #[inline]
    fn from(value: StackVec<T, N>) -> Self {
        Self(InnerVec::Stack(value))
    }
}

/// Creates a `StackVec` containing the arguments.
///
/// The syntax is similar to [`vec!`](https://doc.rust-lang.org/std/macro.vec.html) .
///
/// You must explicitly specify the container capacity.
/// If the input elements exceed the capacity, heap storage will be used instead.
///
/// When called with no arguments, it can be computed at compile time.
/// Otherwise, due to the possibility of creating a [`Vec`], it will be delayed until runtime.
///
/// # Examples
///
/// ```
/// # use fastvec::{autovec, AutoVec};
/// let vec: AutoVec<String, 10> = autovec![];
/// let vec: AutoVec<i64, 10> = autovec![1; 5]; // Need to support Clone.
/// let vec: AutoVec<_, 10> = autovec![1, 2, 3, 4];
/// ```
#[macro_export]
macro_rules! autovec {
    [] => { $crate::AutoVec::new() };
    [$elem:expr; $n:expr] => { $crate::AutoVec::from_elem($elem, $n) };
    [$($item:expr),+ $(,)?] => { $crate::AutoVec::from_buf([ $($item),+ ]) };
}

impl<T, const N: usize> AutoVec<T, N> {
    /// Constructs a new, empty `AutoVec` on the stack with the specified capacity.
    ///
    /// The capacity must be provided at compile time via the const generic parameter.
    ///
    /// Note that the stack memory is allocated when the `AutoVec` is instantiated.
    /// The capacity should not be too large to avoid stack overflow.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::AutoVec;
    /// let mut vec: AutoVec<i32, 8> = AutoVec::new();
    /// ```
    #[inline]
    pub const fn new() -> Self {
        Self(InnerVec::Stack(StackVec::new()))
    }

    /// Modify the stack capacity of the container.
    ///
    /// This function does not move heap data to the stack.
    /// But it is possible to move stack data to the heap (if the capacity is insufficient).
    ///
    /// Unlike [`StackVec::force_cast`], this function will not delete data.
    #[inline]
    pub fn force_cast<const P: usize>(self) -> AutoVec<T, P> {
        match self.0 {
            InnerVec::Stack(mut stack_vec) => {
                let len = stack_vec.len();
                let mut res = <AutoVec<T, P>>::with_capacity(len);
                unsafe {
                    ptr::copy_nonoverlapping(stack_vec.as_ptr(), res.as_mut_ptr(), len);
                    res.set_len(len);
                    stack_vec.set_len(0);
                }
                res
            }
            InnerVec::Heap(items) => AutoVec::<T, P>(InnerVec::Heap(items)),
        }
    }

    /// Return `true` if the data is stored on stack.
    ///
    /// # Example
    ///
    /// ```
    /// # use fastvec::{AutoVec, autovec};
    /// let mut vec: AutoVec<i32, 8> = autovec![];
    ///
    /// assert!(vec.in_stack());
    ///
    /// vec.reserve(10);
    /// assert!(!vec.in_stack());
    /// ```
    #[inline(always)]
    pub const fn in_stack(&self) -> bool {
        match &self.0 {
            InnerVec::Stack(_) => true,
            InnerVec::Heap(_) => false,
        }
    }

    /// Creates an `AutoVec` directly from a pointer, a length, and a capacity.
    ///
    /// This function will create a [`Vec`] internally and store it as heap data.
    /// (Stack storage is bypassed to ensure maximum safety.)
    ///
    /// # Safety
    ///
    /// See more information in [`Vec::from_raw_parts`].
    pub unsafe fn from_raw_parts(ptr: *mut T, length: usize, capacity: usize) -> Self {
        // Safety: See [`Vec::from_raw_parts`].
        unsafe { Vec::from_raw_parts(ptr, length, capacity).into() }
    }

    /// Creates an `AutoVec` from an array.
    ///
    /// If the array length is greater than `N`, heap storage will be used.
    ///
    /// # Panics
    /// Panics if array length > N.
    ///
    /// # Examples
    /// ```
    /// # use fastvec::AutoVec;
    /// let vec: AutoVec<i32, 5> = AutoVec::from_buf([1, 2, 3]);
    /// assert_eq!(vec.len(), 3);
    /// assert!(vec.in_stack());
    /// ```
    #[inline]
    pub fn from_buf<const P: usize>(arr: [T; P]) -> Self {
        let mut vec;

        if P <= N {
            vec = Self(InnerVec::Stack(StackVec::new()));
        } else {
            vec = Self(InnerVec::Heap(Vec::with_capacity(P)));
        }

        unsafe {
            ptr::copy_nonoverlapping(arr.as_ptr(), vec.as_mut_ptr(), P);
            vec.set_len(P);
        }
        core::mem::forget(arr);

        vec
    }

    /// Constructs a new, empty `AutoVec` with at least the specified capacity.
    ///
    /// If the specified capacity is less than or equal to `N`, this is equivalent to [`new`](AutoVec::new),
    /// and no heap memory will be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{AutoVec, autovec};
    /// let mut data_num = 4;
    ///
    /// let vec: AutoVec<i32, 5> = AutoVec::with_capacity(data_num);
    /// assert!(vec.in_stack());
    ///
    /// data_num = 10;
    /// let vec: AutoVec<i32, 5> = AutoVec::with_capacity(data_num);
    /// assert!(!vec.in_stack());
    /// ```
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity > N {
            Self(InnerVec::Heap(Vec::with_capacity(capacity)))
        } else {
            Self(InnerVec::Stack(StackVec::new()))
        }
    }

    /// Forcefully move data to the stack and return mutable reference.
    ///
    /// If the length exceeds the capacity, `truncate` will be called.
    ///
    /// If the data is already in the stack, it won't do anything.
    #[inline]
    pub fn force_to_stack(&mut self) -> &mut StackVec<T, N> {
        if let InnerVec::Heap(vec) = &mut self.0 {
            self.0 = InnerVec::Stack(StackVec::from_vec_truncate(vec));
        }
        match &mut self.0 {
            InnerVec::Stack(vec) => vec,
            _ => unreachable!(),
        }
    }

    /// Forcefully move data to the heap and return mutable reference.
    ///
    /// The allocated memory is precise.
    /// If space needs to be reserved, consider using [`reserve`](AutoVec::reserve).
    ///
    /// If the data is already in the heap, it won't do anything.
    #[inline]
    pub fn force_to_heap(&mut self) -> &mut Vec<T> {
        if let InnerVec::Stack(vec) = &mut self.0 {
            self.0 = InnerVec::Heap(vec.into_vec());
        }
        match &mut self.0 {
            InnerVec::Heap(vec) => vec,
            _ => unreachable!(),
        }
    }

    /// Reserves capacity for at least additional more elements to be inserted in the given `AutoVec`.
    ///
    /// The collection may reserve more space to speculatively avoid frequent reallocations.
    ///
    /// This function may move data.
    /// If the target capacity <= N, this will move the data to the stack.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{AutoVec, autovec};
    /// let mut vec: AutoVec<i32, 8> = autovec![];
    /// vec.reserve(5);
    /// assert!(vec.in_stack());
    ///
    /// vec.reserve(10);
    /// assert!(!vec.in_stack());
    /// assert!(vec.capacity() >= 10);
    /// ```
    pub fn reserve(&mut self, additional: usize) {
        let capacity = self.len() + additional;
        if capacity > N {
            match &mut self.0 {
                InnerVec::Stack(vec) => {
                    // SAFETY: capacity >= len
                    self.0 =
                        InnerVec::Heap(unsafe { vec.into_vec_with_capacity_uncheck(capacity) });
                }
                InnerVec::Heap(vec) => vec.reserve(additional),
            }
        } else {
            match &mut self.0 {
                InnerVec::Stack(_) => return,
                InnerVec::Heap(vec) => {
                    // SAFETY: capacity >= len
                    self.0 = InnerVec::Stack(unsafe { StackVec::from_vec_uncheck(vec) });
                }
            }
        }
    }

    /// Reserves capacity for at least additional more elements to be inserted in the given `AutoVec`.
    ///
    /// This function may move data.
    /// If the target capacity <= N, this will move the data to the stack.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{AutoVec, autovec};
    /// let mut vec: AutoVec<i32, 8> = autovec![];
    /// vec.reserve(5);
    /// assert!(vec.in_stack());
    ///
    /// vec.reserve(10);
    /// assert!(!vec.in_stack());
    /// assert_eq!(vec.capacity(), 10);
    /// ```
    pub fn reserve_exact(&mut self, additional: usize) {
        let capacity = self.len() + additional;
        if capacity > N {
            match &mut self.0 {
                InnerVec::Stack(vec) => {
                    // SAFETY: Ensure that the capacity is greater than the length.
                    self.0 =
                        InnerVec::Heap(unsafe { vec.into_vec_with_capacity_uncheck(capacity) });
                }
                InnerVec::Heap(vec) => vec.reserve_exact(additional),
            }
        } else {
            match &mut self.0 {
                InnerVec::Stack(_) => return,
                InnerVec::Heap(vec) => {
                    // SAFETY: Ensure that the capacity is greater than the length.
                    self.0 = InnerVec::Stack(unsafe { StackVec::from_vec_uncheck(vec) });
                }
            }
        }
    }

    /// Shrinks the capacity of the vector as much as possible.
    ///
    /// If the data is already in the stack, it won't do anything.
    ///
    /// If the capacity is sufficient, this function will move the data to stack.
    pub fn shrink_to_fit(&mut self) {
        match &mut self.0 {
            InnerVec::Heap(vec) => {
                if vec.len() > N {
                    vec.shrink_to_fit();
                } else {
                    // SAFETY: capacity >= len
                    self.0 = InnerVec::Stack(unsafe { StackVec::from_vec_uncheck(vec) });
                }
            }
            InnerVec::Stack(_) => return,
        }
    }

    /// Shrinks the capacity of the vector with a lower bound.
    ///
    /// If the current capacity is less than the lower limit, it's eq to `shrink_to_fit`.
    ///
    /// If the data is already in the stack, it won't do anything.
    ///
    /// If the capacity is sufficient, this function will move the data to stack.
    pub fn shrink_to(&mut self, min_capacity: usize) {
        match &mut self.0 {
            InnerVec::Heap(vec) => {
                if min_capacity.max(vec.len()) > N {
                    vec.shrink_to_fit();
                } else {
                    // SAFETY: capacity >= len
                    self.0 = InnerVec::Stack(unsafe { StackVec::from_vec_uncheck(vec) });
                }
            }
            InnerVec::Stack(_) => return,
        }
    }

    /// Returns a raw pointer to the vector’s buffer, or a dangling raw pointer valid for zero sized reads.
    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        match &self.0 {
            InnerVec::Stack(vec) => vec.as_ptr(),
            InnerVec::Heap(vec) => vec.as_ptr(),
        }
    }

    /// Returns a raw pointer to the vector’s buffer, or a dangling raw pointer valid for zero sized reads.
    #[inline]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        match &mut self.0 {
            InnerVec::Stack(vec) => vec.as_mut_ptr(),
            InnerVec::Heap(vec) => vec.as_mut_ptr(),
        }
    }

    /// Returns the number of elements in the vector, also referred to as its ‘length’.
    #[inline]
    pub const fn len(&self) -> usize {
        match &self.0 {
            InnerVec::Stack(vec) => vec.len(),
            InnerVec::Heap(vec) => vec.len(),
        }
    }

    /// Forces the length of the vector to `new_len`.
    ///
    /// # Safety
    /// - `new_len` needs to be less than or equal to capacity `N`.
    /// - If the length is increased, it is necessary to ensure that the new element is initialized correctly.
    /// - If the length is reduced, it is necessary to ensure that the reduced elements can be dropped normally.
    ///
    /// See [`Vec::set_len`] and [`StackVec::set_len`] .
    #[inline]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        // SAFETY: See function docs.
        unsafe {
            match &mut self.0 {
                InnerVec::Stack(vec) => vec.set_len(new_len),
                InnerVec::Heap(vec) => vec.set_len(new_len),
            }
        }
    }

    /// Returns true if the vector contains no elements.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        match &self.0 {
            InnerVec::Stack(vec) => vec.is_empty(),
            InnerVec::Heap(vec) => vec.is_empty(),
        }
    }

    /// Returns the total number of elements the vector can hold without reallocating.
    ///
    /// For [StackVec], this is always equal to `N` .
    #[inline]
    pub const fn capacity(&self) -> usize {
        match &self.0 {
            InnerVec::Stack(vec) => vec.capacity(),
            InnerVec::Heap(vec) => vec.capacity(),
        }
    }

    /// Convert [`AutoVec`] to [`Vec`].
    ///
    /// If the data is in the stack, the exact memory will be allocated.
    /// If the data is in the heap, will not reallocate memory.
    ///
    /// Therefore, this function is efficient, but the returned [`Vec`] may not be tight.
    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        match self.0 {
            InnerVec::Stack(mut vec) => vec.into_vec(),
            InnerVec::Heap(vec) => vec,
        }
    }

    /// Convert [`AutoVec`] to [`Vec`].
    ///
    /// If the data is in the stack, the exact memory will be allocated.
    /// If the data is in the heap, [`Vec::shrink_to_fit`] will be called.
    #[inline]
    pub fn shrink_into_vec(self) -> Vec<T> {
        match self.0 {
            InnerVec::Stack(mut vec) => vec.into_vec(),
            InnerVec::Heap(mut vec) => {
                vec.shrink_to_fit();
                vec
            }
        }
    }

    /// Forcefully move data to the stack and return mutable reference.
    ///
    /// If the length exceeds the capacity, [`truncate`](StackVec::truncate) will be called.
    ///
    /// If the data is already in the stack, it won't do anything.
    #[inline]
    pub fn truncate_into_stack(mut self) -> StackVec<T, N> {
        if let InnerVec::Heap(vec) = &mut self.0 {
            self.0 = InnerVec::Stack(StackVec::from_vec_truncate(vec));
        }
        match self.0 {
            InnerVec::Stack(vec) => vec,
            _ => unreachable!(),
        }
    }

    /// Convert to internal container.
    #[inline(always)]
    pub fn into_inner(self) -> InnerVec<T, N> {
        self.0
    }

    /// Return the reference of inner vector.
    #[inline(always)]
    pub fn inner_ref(&self) -> InnerVecRef<'_, T, N> {
        match &self.0 {
            InnerVec::Stack(stack_vec) => InnerVecRef::Stack(stack_vec),
            InnerVec::Heap(vec) => InnerVecRef::Heap(vec),
        }
    }

    /// Return the mutable reference of inner vector.
    #[inline(always)]
    pub fn inner_mut(&mut self) -> InnerVecMut<'_, T, N> {
        match &mut self.0 {
            InnerVec::Stack(stack_vec) => InnerVecMut::Stack(stack_vec),
            InnerVec::Heap(vec) => InnerVecMut::Heap(vec),
        }
    }

    /// Converts the [`AutoVec`] into [`Box<[T]>`](Box).
    #[inline]
    pub fn into_boxed_slice(self) -> Box<[T]> {
        match self.0 {
            InnerVec::Stack(mut vec) => vec.into_boxed_slice(),
            InnerVec::Heap(vec) => vec.into_boxed_slice(),
        }
    }

    /// Shortens the vector, keeping the first `len` elements and dropping the rest.
    ///
    /// If `len` is greater or equal to the vector’s current length, this has no effect.
    ///
    /// Note that this will not modify the capacity, so it will not move data.
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        match &mut self.0 {
            InnerVec::Stack(vec) => vec.truncate(len),
            InnerVec::Heap(vec) => vec.truncate(len),
        }
    }

    /// Extracts a slice containing the entire vector.
    #[inline]
    pub const fn as_slice(&self) -> &[T] {
        match &self.0 {
            InnerVec::Stack(vec) => vec.as_slice(),
            InnerVec::Heap(vec) => vec.as_slice(),
        }
    }

    /// Extracts a mutable slice of the entire vector.
    #[inline]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        match &mut self.0 {
            InnerVec::Stack(vec) => vec.as_mut_slice(),
            InnerVec::Heap(vec) => vec.as_mut_slice(),
        }
    }

    /// Removes an element from the vector and returns it.
    ///
    /// The removed element is replaced by the last element of the vector.
    ///
    /// This function does not affect the position (stack/heap) of the data.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    pub fn swap_remove(&mut self, index: usize) -> T {
        match &mut self.0 {
            InnerVec::Stack(vec) => vec.swap_remove(index),
            InnerVec::Heap(vec) => vec.swap_remove(index),
        }
    }

    /// Inserts an element at position `index` within the vector, shifting all elements after it to the right.
    ///
    /// If the heap is insufficient, it will switch to [`Vec`] and reserve some additional memory.
    ///
    /// # Panics
    /// Panics if index > len.
    ///
    /// # Examples
    /// ```
    /// # use fastvec::{AutoVec, autovec};
    /// let mut vec: AutoVec<_, 4> = autovec!['a', 'b', 'c'];
    ///
    /// vec.insert(1, 'd');
    /// assert_eq!(vec, ['a', 'd', 'b', 'c']);
    /// assert!(vec.in_stack());
    ///
    /// vec.insert(4, 'e');
    /// assert_eq!(vec, ['a', 'd', 'b', 'c', 'e']);
    /// assert!(!vec.in_stack());
    /// ```
    #[inline]
    pub fn insert(&mut self, index: usize, element: T) {
        match &mut self.0 {
            InnerVec::Stack(vec) => {
                assert!(index <= vec.len(), "insertion index should be <= len");
                if vec.len() < N {
                    // SAFETY: index <= len && len < N
                    unsafe {
                        vec.insert_uncheck(index, element);
                    }
                } else {
                    let mut new_vec: Vec<T> = Vec::with_capacity(N + { N >> 1 } + 4);
                    let dst_ptr = new_vec.as_mut_ptr();
                    let src_ptr = vec.as_ptr();
                    // SAFETY: enough capacity and valid data.
                    unsafe {
                        ptr::copy_nonoverlapping(src_ptr, dst_ptr, index);
                        ptr::write(dst_ptr.add(index), element);
                        ptr::copy_nonoverlapping(
                            src_ptr.add(index),
                            dst_ptr.add(index + 1),
                            N - index,
                        );
                        vec.set_len(0);
                        new_vec.set_len(N + 1);
                    }
                    self.0 = InnerVec::Heap(new_vec);
                }
            }
            InnerVec::Heap(vec) => vec.insert(index, element),
        }
    }

    /// Removes and returns the element at position index within the vector, shifting all elements after it to the left.
    ///
    /// Note: Because this shifts over the remaining elements, it has a worst-case performance of O(n).
    /// If you don’t need the order of elements to be preserved, use [`swap_remove`](AutoVec::swap_remove) instead.
    ///
    /// This function does not affect the position (stack/heap) of the data.
    ///
    /// # Panics
    /// Panics if `index` is out of bounds.
    ///
    /// # Examples
    /// ```
    /// # use fastvec::{AutoVec, autovec};
    /// let mut v: AutoVec<_, 4> = autovec!['a', 'b', 'c'];
    /// assert_eq!(v.remove(1), 'b');
    /// assert_eq!(v, ['a', 'c']);
    /// ```
    pub fn remove(&mut self, index: usize) -> T {
        match &mut self.0 {
            InnerVec::Stack(vec) => vec.remove(index),
            InnerVec::Heap(vec) => vec.remove(index),
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// This function does not affect the position (stack/heap) of the data.
    #[inline]
    pub fn retain<F: FnMut(&T) -> bool>(&mut self, f: F) {
        match &mut self.0 {
            InnerVec::Stack(vec) => vec.retain(f),
            InnerVec::Heap(vec) => vec.retain(f),
        }
    }

    /// Retains only the elements specified by the predicate, passing a mutable reference to it.
    #[inline]
    pub fn retain_mut<F: FnMut(&mut T) -> bool>(&mut self, f: F) {
        match &mut self.0 {
            InnerVec::Stack(vec) => vec.retain_mut(f),
            InnerVec::Heap(vec) => vec.retain_mut(f),
        }
    }

    /// Removes all but the first of consecutive elements in the vector that resolve to the same key.
    #[inline]
    pub fn dedup_by_key<F, K>(&mut self, key: F)
    where
        F: FnMut(&mut T) -> K,
        K: PartialEq,
    {
        match &mut self.0 {
            InnerVec::Stack(vec) => vec.dedup_by_key(key),
            InnerVec::Heap(vec) => vec.dedup_by_key(key),
        }
    }

    /// Removes all but the first of consecutive elements in the vector satisfying a given equality relation.
    #[inline]
    pub fn dedup_by<F: FnMut(&mut T, &mut T) -> bool>(&mut self, same_bucket: F) {
        match &mut self.0 {
            InnerVec::Stack(vec) => vec.dedup_by(same_bucket),
            InnerVec::Heap(vec) => vec.dedup_by(same_bucket),
        }
    }

    /// Appends an element to the back of a collection.
    ///
    /// If the heap is insufficient, it will switch to [`Vec`] and reserve some additional memory.
    ///
    /// # Time complexity
    /// Takes amortized O(1) time. If the vector’s length would exceed its capacity after the push,
    /// O(capacity) time is taken to copy the vector’s elements to a larger allocation.
    ///
    /// # Examples
    /// ```
    /// # use fastvec::{AutoVec, autovec};
    /// let mut vec: AutoVec<_, 4> = autovec![1, 2];
    /// vec.push(3);
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    #[inline]
    pub fn push(&mut self, value: T) {
        match &mut self.0 {
            InnerVec::Stack(vec) => {
                if vec.len() < N {
                    // SAFETY: len < N
                    unsafe { vec.push_uncheck(value) };
                } else {
                    let mut new_vec: Vec<T> = Vec::with_capacity(N + { N >> 1 } + 4);
                    let dst_ptr = new_vec.as_mut_ptr();
                    let src_ptr = vec.as_ptr();
                    // SAFETY: enough capacity and valid data.
                    unsafe {
                        ptr::copy_nonoverlapping(src_ptr, dst_ptr, N);
                        ptr::write(dst_ptr.add(N), value);
                        vec.set_len(0);
                        new_vec.set_len(N + 1);
                    }
                    self.0 = InnerVec::Heap(new_vec);
                }
            }
            InnerVec::Heap(vec) => vec.push(value),
        }
    }

    /// Removes the last element from a vector and returns it, or None if it is empty.
    ///
    /// This function does not affect the position (stack/heap) of the data.
    ///
    /// # Time complexity
    /// O(1) time
    ///
    /// # Examples
    /// ```
    /// # use fastvec::{AutoVec, autovec};
    /// let mut vec: AutoVec<_, 4> = autovec![1, 2, 3];
    /// assert_eq!(vec.pop(), Some(3));
    /// assert_eq!(vec, [1, 2]);
    /// ```
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        match &mut self.0 {
            InnerVec::Stack(vec) => vec.pop(),
            InnerVec::Heap(vec) => vec.pop(),
        }
    }

    /// Removes and returns the last element from a vector if the predicate returns `true`,
    /// or `None` if the predicate returns false or the vector is empty (the predicate will not be called in that case).
    #[inline]
    pub fn pop_if(&mut self, predicate: impl FnOnce(&mut T) -> bool) -> Option<T> {
        match &mut self.0 {
            InnerVec::Stack(vec) => vec.pop_if(predicate),
            InnerVec::Heap(vec) => vec.pop_if(predicate),
        }
    }

    /// Moves all the elements of other into self, leaving other empty.
    ///
    /// This function does not affect the capacity of **other**,
    /// so not affect the position (stack/heap) of the other data.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{AutoVec, autovec};
    /// let mut vec1: AutoVec<_, 4> = autovec![1, 2, 3];
    /// let mut vec2: AutoVec<_, 4> = autovec![4, 5, 6];
    /// vec1.append(&mut vec2);
    ///
    /// assert_eq!(vec1, [1, 2, 3, 4, 5, 6]);
    /// assert_eq!(vec2, []);
    /// # assert!(!vec1.in_stack());
    /// # assert!(vec2.in_stack());
    /// ```
    #[inline]
    pub fn append<const P: usize>(&mut self, other: &mut AutoVec<T, P>) {
        match &mut other.0 {
            InnerVec::Stack(vec) => {
                self.append_stack_vec(vec);
            }
            InnerVec::Heap(vec) => {
                self.append_vec(vec);
            }
        }
    }

    /// Moves all the elements of [`StackVec`] into self, leaving [`StackVec`] empty.
    pub fn append_stack_vec<const P: usize>(&mut self, other: &mut StackVec<T, P>) {
        match &mut self.0 {
            InnerVec::Stack(vec) => {
                if vec.len() + other.len() > N {
                    let mut vec = vec.into_vec_with_capacity(vec.len() + other.len());
                    other.append_to_vec(&mut vec);
                    self.0 = InnerVec::Heap(vec);
                } else {
                    vec.append(other);
                }
            }
            InnerVec::Heap(vec) => {
                other.append_to_vec(vec);
            }
        }
    }

    /// Moves all the elements of [`Vec`] into self, leaving [`Vec`] empty.
    pub fn append_vec(&mut self, other: &mut Vec<T>) {
        match &mut self.0 {
            InnerVec::Stack(vec) => {
                if vec.len() + other.len() > N {
                    let mut vec = vec.into_vec_with_capacity(vec.len() + other.len());
                    vec.append(other);
                    self.0 = InnerVec::Heap(vec);
                } else {
                    vec.append_vec(other);
                }
            }
            InnerVec::Heap(vec) => {
                vec.append(other);
            }
        }
    }

    /// Clears the vector, removing all values.
    ///
    /// Note that this method has no effect on the allocated capacity of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{AutoVec, autovec};
    /// let mut vec1: AutoVec<_, 4> = autovec![1, 2, 3];
    /// vec1.clear();
    /// assert!(vec1.is_empty());
    ///
    /// let mut vec1: AutoVec<_, 4> = autovec![1, 2, 3, 4, 5];
    /// assert!(!vec1.in_stack());
    /// vec1.clear();
    /// assert!(!vec1.in_stack());
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        match &mut self.0 {
            InnerVec::Stack(vec) => vec.clear(),
            InnerVec::Heap(vec) => vec.clear(),
        }
    }

    /// Splits the collection into two at the given index.
    ///
    /// # Panics
    /// Panics if at > len.
    ///
    /// # Examples
    /// ```
    /// # use fastvec::{AutoVec, autovec};
    /// let mut vec : AutoVec<_, 4> = autovec!['a', 'b', 'c'];
    /// let vec2 = vec.split_off(1);
    ///
    /// assert_eq!(vec.as_slice(), ['a']);
    /// assert_eq!(vec2.as_slice(), ['b', 'c']);
    /// ```
    #[inline]
    pub fn split_off(&mut self, at: usize) -> Self {
        match &mut self.0 {
            InnerVec::Stack(vec) => vec.split_off(at).into(),
            InnerVec::Heap(vec) => vec.split_off(at).into(),
        }
    }

    /// Resizes the Vec in-place so that len is equal to new_len.
    ///
    /// This function may move data from the stack to the heap (if capacity is insufficient),
    /// but it will not move data from the heap to the stack.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{AutoVec, autovec};
    /// let mut vec : AutoVec<_, 4> = autovec![1, 2, 3];
    /// vec.resize_with(5, Default::default);
    /// assert_eq!(vec.as_slice(), [1, 2, 3, 0, 0]);
    /// assert!(!vec.in_stack());
    ///
    /// let mut vec : AutoVec<_, 4> = autovec![];
    /// let mut p = 1;
    /// vec.resize_with(4, || { p *= 2; p });
    /// assert_eq!(vec.as_slice(), [2, 4, 8, 16]);
    /// assert!(vec.in_stack());
    ///
    /// let mut vec : AutoVec<_, 3> = autovec![1, 2, 3, 4, 5];
    /// vec.resize_with(2, Default::default);
    /// assert!(!vec.in_stack());
    /// ```
    #[inline]
    pub fn resize_with<F: FnMut() -> T>(&mut self, new_len: usize, f: F) {
        match &mut self.0 {
            InnerVec::Stack(vec) => {
                if new_len <= N {
                    vec.resize_with(new_len, f);
                } else {
                    // SAFETY: capacity = new_len > len
                    let mut vec = unsafe { vec.into_vec_with_capacity_uncheck(new_len) };
                    vec.resize_with(new_len, f);
                    self.0 = InnerVec::Heap(vec);
                }
            }
            InnerVec::Heap(vec) => vec.resize_with(new_len, f),
        }
    }

    /// Consumes and leaks the [`AutoVec`], returning a mutable reference to the contents, `&'a mut [T]`.
    ///
    /// This will pre transfer the data to the heap to ensure the validity of the references.
    #[inline]
    pub fn leak<'a>(self) -> &'a mut [T] {
        self.into_vec().leak()
    }

    /// Returns the remaining spare capacity of the vector as a slice of `MaybeUninit<T>`.
    ///
    /// The returned slice can be used to fill the vector with data (e.g. by reading from a file)
    /// before marking the data as initialized using the [`set_len`](AutoVec::set_len) method.
    ///
    /// See [`Vec::spare_capacity_mut`] and [`StackVec::spare_capacity_mut`] .
    #[inline]
    pub fn spare_capacity_mut(&mut self) -> &mut [core::mem::MaybeUninit<T>] {
        match &mut self.0 {
            InnerVec::Stack(vec) => vec.spare_capacity_mut(),
            InnerVec::Heap(vec) => vec.spare_capacity_mut(),
        }
    }
}

impl<T: Clone, const N: usize> AutoVec<T, N> {
    /// Creates an `AutoVec` with `num` copies of `elem`.
    ///
    /// If `num > N`, heap storage will be used.
    ///
    /// This function requires `T` to implement `Clone`.
    ///
    /// # Examples
    /// ```
    /// # use fastvec::AutoVec;
    /// let vec: AutoVec<i32, 5> = AutoVec::from_elem(1, 4);
    /// assert_eq!(vec.len(), 4);
    /// assert!(vec.in_stack());
    /// ```
    #[inline]
    pub fn from_elem(elem: T, num: usize) -> Self {
        let mut vec;
        if num <= N {
            vec = Self(InnerVec::Stack(StackVec::new()));
        } else {
            vec = Self(InnerVec::Heap(Vec::with_capacity(num)));
        }
        let base_ptr = vec.as_mut_ptr();
        let mut cnt = 1;
        unsafe {
            while cnt < num {
                ptr::write(base_ptr.add(cnt), elem.clone());
                cnt += 1;
            }
            if num != 0 {
                // Reduce one copy.
                ptr::write(base_ptr, elem);
            }
            vec.set_len(num);
        }
        vec
    }

    /// Resizes the [`AutoVec`] in-place so that len is equal to `new_len`.
    ///
    /// If `new_len` is greater than `len`, the [`AutoVec`] is extended by the difference,
    /// with each additional slot filled with value. If `new_len` is less than `len`, the [`AutoVec`] is simply truncated.
    ///
    /// This function may move data from the stack to the heap (if capacity is insufficient),
    /// but it will not move data from the heap to the stack.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{AutoVec, autovec};
    /// let mut vec: AutoVec<_, 5> = autovec!["hello"];
    /// vec.resize(3, "world");
    /// assert_eq!(vec.as_slice(), ["hello", "world", "world"]);
    ///
    /// let mut vec: AutoVec<_, 5> = autovec!['a', 'b', 'c', 'd'];
    /// vec.resize(2, '_');
    /// assert_eq!(vec.as_slice(), ['a', 'b']);
    /// ```
    #[inline]
    pub fn resize(&mut self, new_len: usize, value: T) {
        match &mut self.0 {
            InnerVec::Stack(vec) => {
                if new_len <= N {
                    vec.resize(new_len, value);
                } else {
                    // SAFETY: capacity == new_len > len
                    let mut vec = unsafe { vec.into_vec_with_capacity_uncheck(new_len) };
                    vec.resize(new_len, value);
                    self.0 = InnerVec::Heap(vec);
                }
            }
            InnerVec::Heap(vec) => vec.resize(new_len, value),
        }
    }

    /// Clones and appends all elements in a slice to the [`AutoVec`].
    ///
    /// # Examples
    /// ```
    /// # use fastvec::{AutoVec, autovec};
    /// let mut vec: AutoVec<_, 5> = autovec![1];
    /// vec.extend_from_slice(&[2, 3, 4]);
    /// assert_eq!(vec, [1, 2, 3, 4]);
    /// ```
    #[inline]
    pub fn extend_from_slice(&mut self, other: &[T]) {
        match &mut self.0 {
            InnerVec::Stack(vec) => {
                let capacity = vec.len() + other.len();
                if capacity <= N {
                    vec.extend_from_slice(other);
                } else {
                    // SAFETY: capacity == new_len > len
                    let mut vec = unsafe { vec.into_vec_with_capacity_uncheck(capacity) };
                    vec.extend_from_slice(other);
                    self.0 = InnerVec::Heap(vec);
                }
            }
            InnerVec::Heap(vec) => vec.extend_from_slice(other),
        }
    }

    /// Given a range src, clones a slice of elements in that range and appends it to the end..
    #[inline]
    pub fn extend_from_within<R: core::ops::RangeBounds<usize>>(&mut self, src: R) {
        match &mut self.0 {
            InnerVec::Stack(vec) => {
                let (start, end) = crate::utils::split_range_bound(&src, vec.len());
                let capacity = end - start + vec.len();
                if capacity <= N {
                    vec.extend_from_within(src);
                } else {
                    // SAFETY: capacity == new_len > len
                    let mut vec = unsafe { vec.into_vec_with_capacity_uncheck(capacity) };
                    vec.extend_from_within(src);
                    self.0 = InnerVec::Heap(vec);
                }
            }
            InnerVec::Heap(vec) => vec.extend_from_within(src),
        }
    }
}

impl<T: PartialEq, const N: usize> AutoVec<T, N> {
    /// Removes consecutive repeated elements in the vector according to the PartialEq trait implementation.
    #[inline]
    pub fn dedup(&mut self) {
        match &mut self.0 {
            InnerVec::Stack(vec) => vec.dedup(),
            InnerVec::Heap(vec) => vec.dedup(),
        }
    }
}

impl<T, const N: usize, const P: usize> AutoVec<[T; P], N> {
    /// Takes a `AutoVec<[T; P], N>` and flattens it into a `AutoVec<T, S>`.
    ///
    /// If the capacity is insufficient, [`Vec`] will be used.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{AutoVec, autovec};
    /// let mut vec: AutoVec<_, 2> = autovec![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    /// assert_eq!(vec.pop(), Some([7, 8, 9]));
    /// assert!(!vec.in_stack());
    ///
    /// let mut flattened = vec.into_flattened::<6>();
    /// assert_eq!(flattened.as_slice(), [1, 2, 3, 4, 5, 6]);
    /// assert!(flattened.in_stack());
    ///
    /// let mut vec: AutoVec<_, 3> = autovec![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    /// assert_eq!(vec.pop(), Some([7, 8, 9]));
    /// assert!(vec.in_stack());
    ///
    /// let mut flattened = vec.into_flattened::<5>();
    /// assert_eq!(flattened.as_slice(), [1, 2, 3, 4, 5, 6]);
    /// assert!(!flattened.in_stack());
    /// ```
    #[inline]
    pub fn into_flattened<const S: usize>(self) -> AutoVec<T, S> {
        match self.0 {
            InnerVec::Stack(mut vec) => {
                if S >= P * vec.len() {
                    vec.into_flattened::<S>().into()
                } else {
                    vec.into_vec().into_flattened().into()
                }
            }
            InnerVec::Heap(vec) => {
                if S >= P * vec.len() {
                    // SAFETY: capasity == S > new_len == P * old_len
                    unsafe { <StackVec<T, S>>::from_vec_uncheck(&mut vec.into_flattened()).into() }
                } else {
                    vec.into_flattened().into()
                }
            }
        }
    }
}

impl<T, const N: usize> Default for AutoVec<T, N> {
    /// Constructs a new, empty [`AutoVec`] on the stack with the specified capacity.
    ///
    /// It's eq to [`AutoVec::new`] .
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T: 'a + Clone, const N: usize> Extend<&'a T> for AutoVec<T, N> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let (hint, _) = iter.size_hint();
        self.reserve(hint);

        for item in iter {
            self.push(item.clone());
        }
    }
}

impl<T, const N: usize> Extend<T> for AutoVec<T, N> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let (hint, _) = iter.size_hint();
        self.reserve(hint);
        for item in iter {
            self.push(item);
        }
    }
}

crate::utils::impl_commen_traits!(AutoVec<T, N>);

impl<T, U, const N: usize> core::cmp::PartialEq<AutoVec<U, N>> for AutoVec<T, N>
where
    T: core::cmp::PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &AutoVec<U, N>) -> bool {
        core::cmp::PartialEq::eq(self.as_slice(), other.as_slice())
    }
}

impl<'a, T: Clone, const N: usize> From<&'a AutoVec<T, N>> for alloc::borrow::Cow<'a, [T]> {
    fn from(v: &'a AutoVec<T, N>) -> alloc::borrow::Cow<'a, [T]> {
        alloc::borrow::Cow::Borrowed(v.as_slice())
    }
}

impl<'a, T: Clone, const N: usize> From<AutoVec<T, N>> for alloc::borrow::Cow<'a, [T]> {
    fn from(v: AutoVec<T, N>) -> alloc::borrow::Cow<'a, [T]> {
        alloc::borrow::Cow::Owned(v.into_vec())
    }
}

impl<T: Clone, const N: usize> From<&[T]> for AutoVec<T, N> {
    fn from(value: &[T]) -> Self {
        if value.len() <= N {
            <StackVec<T, N> as From<&[T]>>::from(value).into()
        } else {
            <Vec<T> as From<&[T]>>::from(value).into()
        }
    }
}

impl<T: Clone, const N: usize, const P: usize> From<&[T; P]> for AutoVec<T, N> {
    fn from(value: &[T; P]) -> Self {
        if P <= N {
            <StackVec<T, N> as From<&[T; P]>>::from(value).into()
        } else {
            <Vec<T> as From<&[T; P]>>::from(value).into()
        }
    }
}

impl<T: Clone, const N: usize> From<&mut [T]> for AutoVec<T, N> {
    #[inline]
    fn from(value: &mut [T]) -> Self {
        <Self as From<&[T]>>::from(value)
    }
}

impl<T: Clone, const N: usize, const P: usize> From<&mut [T; P]> for AutoVec<T, N> {
    #[inline]
    fn from(value: &mut [T; P]) -> Self {
        <Self as From<&[T; P]>>::from(value)
    }
}

impl<T, const N: usize, const P: usize> From<[T; P]> for AutoVec<T, N> {
    fn from(value: [T; P]) -> Self {
        if P <= N {
            <StackVec<T, N> as From<[T; P]>>::from(value).into()
        } else {
            <Vec<T> as From<[T; P]>>::from(value).into()
        }
    }
}

impl<T, const N: usize> From<Box<[T]>> for AutoVec<T, N> {
    fn from(value: Box<[T]>) -> Self {
        if value.len() <= N {
            <StackVec<T, N> as From<Box<[T]>>>::from(value).into()
        } else {
            <Vec<T> as From<Box<[T]>>>::from(value).into()
        }
    }
}

impl<T, const N: usize> FromIterator<T> for AutoVec<T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (hint, _) = iter.size_hint();

        let mut vec = if hint <= N {
            Self::new()
        } else {
            Vec::with_capacity(hint).into()
        };

        for item in iter {
            vec.push(item);
        }
        vec
    }
}

impl<T, const N: usize> IntoIterator for AutoVec<T, N> {
    type Item = T;
    type IntoIter = IntoIter<T, N>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        match self.0 {
            InnerVec::Stack(vec) => IntoIter::Stack(IntoIterator::into_iter(vec)),
            InnerVec::Heap(vec) => IntoIter::Heap(IntoIterator::into_iter(vec)),
        }
    }
}

/// An iterator that consumes a [`AutoVec`] and yields its items by value.
#[derive(Clone)]
pub enum IntoIter<T, const N: usize> {
    Stack(crate::stack_vec::IntoIter<T, N>),
    Heap(alloc::vec::IntoIter<T>),
}

impl<T, const N: usize> IntoIter<T, N> {
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        match self {
            IntoIter::Stack(iter) => iter.as_slice(),
            IntoIter::Heap(iter) => iter.as_slice(),
        }
    }
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            IntoIter::Stack(iter) => iter.as_mut_slice(),
            IntoIter::Heap(iter) => iter.as_mut_slice(),
        }
    }
}

impl<T, const N: usize> Iterator for IntoIter<T, N> {
    type Item = T;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            IntoIter::Stack(iter) => Iterator::next(iter),
            IntoIter::Heap(iter) => Iterator::next(iter),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            IntoIter::Stack(iter) => Iterator::size_hint(iter),
            IntoIter::Heap(iter) => Iterator::size_hint(iter),
        }
    }
}

impl<T, const N: usize> DoubleEndedIterator for IntoIter<T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            IntoIter::Stack(iter) => DoubleEndedIterator::next_back(iter),
            IntoIter::Heap(iter) => DoubleEndedIterator::next_back(iter),
        }
    }
}

impl<T, const N: usize> ExactSizeIterator for IntoIter<T, N> {}

impl<T, const N: usize> FusedIterator for IntoIter<T, N> {}

impl<T, const N: usize> Default for IntoIter<T, N> {
    fn default() -> Self {
        Self::Stack(crate::stack_vec::IntoIter::default())
    }
}

impl<T: fmt::Debug, const N: usize> fmt::Debug for IntoIter<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("IntoIter").field(&self.as_slice()).finish()
    }
}

/// An iterator that removes the items from a [`AutoVec`] and yields them by value.
///
/// See [`AutoVec::drain`] .
pub enum Drain<'a, T: 'a, const N: usize> {
    Stack(crate::stack_vec::Drain<'a, T, N>),
    Heap(alloc::vec::Drain<'a, T>),
}

impl<T, const N: usize> AutoVec<T, N> {
    /// Removes the subslice indicated by the given range from the vector,
    /// returning a double-ended iterator over the removed subslice.
    ///
    /// # Panics
    /// Panics if the range has `start_bound > end_bound`, or,
    /// if the range is bounded on either end and past the length of the vector.
    ///
    /// # Leaking
    /// If the returned iterator goes out of scope without being dropped (due to [`core::mem::forget`], for example),
    /// the vector may have lost and leaked elements arbitrarily, including elements outside the range.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{AutoVec, autovec};
    /// let mut v: AutoVec<_, 4> = autovec![1, 2, 3];
    /// let u: Vec<_> = v.drain(1..).collect();
    /// assert_eq!(v.as_slice(), [1]);
    /// assert_eq!(u, [2, 3]);
    ///
    /// v.drain(..);
    /// assert_eq!(v.as_slice(), &[]);
    /// ```
    #[inline]
    pub fn drain<R: core::ops::RangeBounds<usize>>(&mut self, range: R) -> Drain<'_, T, N> {
        match &mut self.0 {
            InnerVec::Stack(vec) => Drain::Stack(vec.drain(range)),
            InnerVec::Heap(vec) => Drain::Heap(vec.drain(range)),
        }
    }
}

impl<T, const N: usize> Drain<'_, T, N> {
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        match &self {
            Drain::Stack(drain) => drain.as_slice(),
            Drain::Heap(drain) => drain.as_slice(),
        }
    }
}

impl<T, const N: usize> AsRef<[T]> for Drain<'_, T, N> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        match &self {
            Drain::Stack(drain) => drain.as_ref(),
            Drain::Heap(drain) => drain.as_ref(),
        }
    }
}

impl<T: fmt::Debug, const N: usize> fmt::Debug for Drain<'_, T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Drain").field(&self.as_slice()).finish()
    }
}

impl<T, const N: usize> Iterator for Drain<'_, T, N> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        match self {
            Drain::Stack(drain) => drain.next(),
            Drain::Heap(drain) => drain.next(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Drain::Stack(drain) => drain.size_hint(),
            Drain::Heap(drain) => drain.size_hint(),
        }
    }
}

impl<T, const N: usize> DoubleEndedIterator for Drain<'_, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        match self {
            Drain::Stack(drain) => drain.next_back(),
            Drain::Heap(drain) => drain.next_back(),
        }
    }
}

impl<T, const N: usize> ExactSizeIterator for Drain<'_, T, N> {
    #[inline]
    fn len(&self) -> usize {
        match self {
            Drain::Stack(drain) => drain.len(),
            Drain::Heap(drain) => drain.len(),
        }
    }
}

impl<T, const N: usize> FusedIterator for Drain<'_, T, N> {}

/// A splicing iterator for [`AutoVec`].
///
/// See [`AutoVec::splice`] .
pub enum Splice<'a, I: ExactSizeIterator + 'a, const N: usize> {
    Stack(crate::stack_vec::Splice<'a, I, N>),
    Heap(alloc::vec::Splice<'a, I>),
}

impl<T, const N: usize> AutoVec<T, N> {
    /// Creates a splicing iterator that replaces the specified range in the vector
    /// with the given `replace_with` iterator and yields the removed items.
    /// `replace_with` does not need to be the same length as `range`.
    ///
    /// See more infomation in [`alloc::vec::Splice`], the only difference is that
    /// we require the `replace_with` is [`ExactSizeIterator`]`.
    ///
    /// This is optimal if:
    ///
    /// * The tail (elements in the vector after `range`) is empty,
    /// * or `replace_with` yields fewer or equal elements than `range`'s length
    ///
    /// If the capacity is insufficient, this will first switch to the heap and then call splice.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{autovec, AutoVec};
    /// let mut v: AutoVec<_, 4> = autovec![1, 2, 3, 4];
    /// let new = [7, 8, 9];
    /// let u: Vec<_> = v.splice(1..3, new).collect();
    ///
    /// assert!(!v.in_stack());
    /// assert_eq!(v, [1, 7, 8, 9, 4]);
    /// assert_eq!(u, [2, 3]);
    /// ```
    ///
    /// Using `splice` to insert new items into a vector efficiently at a specific position
    /// indicated by an empty range:
    ///
    /// ```
    /// # use fastvec::{autovec, AutoVec};
    /// let mut v: AutoVec<_, 5> = autovec![1, 5];
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
        let iter = replace_with.into_iter();
        let len = self.len();
        let (start, end) = crate::utils::split_range_bound(&range, len);
        let capacity = len - end + start + iter.len();

        if capacity > N {
            Splice::Heap(self.force_to_heap().splice(range, iter))
        } else {
            match &mut self.0 {
                InnerVec::Stack(vec) => Splice::Stack(vec.splice(range, iter)),
                InnerVec::Heap(vec) => Splice::Heap(vec.splice(range, iter)),
            }
        }
    }
}

impl<I: ExactSizeIterator, const N: usize> Iterator for Splice<'_, I, N> {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Splice::Stack(splice) => splice.next(),
            Splice::Heap(splice) => splice.next(),
        }
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Splice::Stack(splice) => splice.size_hint(),
            Splice::Heap(splice) => splice.size_hint(),
        }
    }
}

impl<I: ExactSizeIterator, const N: usize> DoubleEndedIterator for Splice<'_, I, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            Splice::Stack(splice) => splice.next_back(),
            Splice::Heap(splice) => splice.next_back(),
        }
    }
}

impl<I: ExactSizeIterator, const N: usize> ExactSizeIterator for Splice<'_, I, N> {
    #[inline]
    fn len(&self) -> usize {
        match self {
            Splice::Stack(splice) => splice.len(),
            Splice::Heap(splice) => splice.len(),
        }
    }
}

impl<I: fmt::Debug + ExactSizeIterator, const N: usize> fmt::Debug for Splice<'_, I, N>
where
    I::Item: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stack(splice) => fmt::Debug::fmt(splice, f),
            Self::Heap(splice) => fmt::Debug::fmt(splice, f),
        }
    }
}

/// An iterator which uses a closure to determine if an element should be removed.
///
/// See [`AutoVec::extract_if`] .
pub enum ExtractIf<'a, T, F: FnMut(&mut T) -> bool, const N: usize> {
    Stack(crate::stack_vec::ExtractIf<'a, T, F, N>),
    Heap(alloc::vec::ExtractIf<'a, T, F>),
}

impl<T, const N: usize> AutoVec<T, N> {
    /// Creates an iterator which uses a closure to determine if an element in the range should be removed.
    ///
    /// See more infomation in [`Vec::extract_if`] .
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
    /// # use fastvec::{autovec, AutoVec};
    /// let mut numbers: AutoVec<_, 20> = autovec![1, 2, 3, 4, 5, 6, 8, 9, 11, 13, 14, 15];
    ///
    /// let evens = numbers.extract_if(.., |x| *x % 2 == 0).collect::<AutoVec<_, 10>>();
    /// let odds = numbers;
    ///
    /// assert_eq!(evens, [2, 4, 6, 8, 14]);
    /// assert_eq!(odds, [1, 3, 5, 9, 11, 13, 15]);
    /// ```
    ///
    /// Using the range argument to only process a part of the vector:
    ///
    /// ```
    /// # use fastvec::{autovec, AutoVec};
    /// let mut items: AutoVec<_, 15> = autovec![0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2];
    /// let ones = items.extract_if(7.., |x| *x == 1).collect::<Vec<_>>();
    /// assert_eq!(items, [0, 0, 0, 0, 0, 0, 0, 2, 2, 2]);
    /// assert_eq!(ones.len(), 3);
    /// ```
    pub fn extract_if<F, R>(&mut self, range: R, filter: F) -> ExtractIf<'_, T, F, N>
    where
        F: FnMut(&mut T) -> bool,
        R: core::ops::RangeBounds<usize>,
    {
        match &mut self.0 {
            InnerVec::Stack(vec) => ExtractIf::Stack(vec.extract_if(range, filter)),
            InnerVec::Heap(vec) => ExtractIf::Heap(vec.extract_if(range, filter)),
        }
    }
}

impl<T, F: FnMut(&mut T) -> bool, const N: usize> Iterator for ExtractIf<'_, T, F, N> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ExtractIf::Stack(extract_if) => extract_if.next(),
            ExtractIf::Heap(extract_if) => extract_if.next(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            ExtractIf::Stack(extract_if) => extract_if.size_hint(),
            ExtractIf::Heap(extract_if) => extract_if.size_hint(),
        }
    }
}

impl<T: fmt::Debug, F: FnMut(&mut T) -> bool, const N: usize> fmt::Debug
    for ExtractIf<'_, T, F, N>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExtractIf::Stack(extract_if) => fmt::Debug::fmt(extract_if, f),
            ExtractIf::Heap(extract_if) => fmt::Debug::fmt(extract_if, f),
        }
    }
}
