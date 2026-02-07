use alloc::{
    alloc::{Layout, alloc, dealloc},
    boxed::Box,
    vec::Vec,
};
use core::{
    cell::Cell,
    fmt,
    mem::{self, MaybeUninit},
    panic::RefUnwindSafe,
    ptr, slice,
};

use crate::utils::{IsZST, cold_path};

/// A type used to manipulate internal data of [`FastVec`](crate::FastVec).
///
/// This type may contain self-references. After moves, the pointer must be refreshed to remain valid,
/// so [`FastVecData::refresh`] should run before each method to ensure correctness.
///
/// Calling `refresh` adds a branch and assignment, and it is unsafe to require callers to do it manually.
///
/// Instead, all constructors are hidden behind the [`FastVec`](crate::FastVec) wrapper.
/// Use [`data`](crate::FastVec::data) to obtain [`&FastVecData`](FastVecData);
/// these entry points refresh automatically.
///
/// During the reference's lifetime the data will not move, so subsequent operations are safe.
///
/// # Examples
///
/// ```
/// # use fastvec::{FastVec, fast_vec::FastVecData};
/// let mut state: FastVec<i32> = [1, 2, 3, 4].into();
/// let vec = state.data();
///
/// vec.push(5);
/// vec.push(6);
/// assert_eq!(vec, &[1, 2, 3,  4, 5, 6]);
/// ```
///
/// Almost all methods supported by [`alloc::vec::Vec`] can be used in [`FastVecData`],
/// As long as its input is a reference to vector self.
///
/// ```
/// # use fastvec::FastVec;
/// let mut state: FastVec<i32, 8> = [1, 2, 3, 4].into();
/// let vec = state.data();
///
/// assert_eq!(vec.capacity(), 8);
/// assert_eq!(vec.len(), 4);
///
/// vec.insert(0, 0);
/// vec.retain(|v| *v % 2 == 0);
///
/// assert_eq!(vec, &[0, 2, 4]);
/// ```
///
/// # Internal Requirements
///
/// These requirements are guaranteed by the implementation; users typically do not need to consider them.
///
/// 1. This type cannot be constructed directly; obtain references via [`FastVec`](crate::FastVec).
/// 2. Calling any method (except `len`, `capacity`, and `in_stack`) requires the internal pointer to be valid; this is
///    usually guaranteed by obtaining a handle with [`FastVec::get`](crate::FastVec::get).
/// 3. Heap allocation is allowed even when `capacity <= N`.
/// 4. If resources are allocated on the heap and `T` is not ZST, the capacity must be non-zero.
pub struct FastVecData<T, const N: usize> {
    pub(crate) cache: [MaybeUninit<T>; N],
    /// We need to use [`Cell`] or [`UnsafeCell`](core::cell::UnsafeCell) to implement internal variability,
    /// When self implemented, [`refresh`](FastVecData::refresh) may be considered useless and optimized.
    pub(crate) ptr: Cell<*mut T>,
    pub(crate) len: usize,
    pub(crate) cap: usize,
    pub(crate) in_stack: bool,
}

unsafe impl<T, const N: usize> Send for FastVecData<T, N> where T: Send {}
unsafe impl<T, const N: usize> Sync for FastVecData<T, N> where T: Sync {}
impl<T, const N: usize> RefUnwindSafe for FastVecData<T, N> where T: RefUnwindSafe {}

impl<T, const N: usize> Drop for FastVecData<T, N> {
    // Internal data using `MaybeUninit`, we need to call `drop` manually.
    fn drop(&mut self) {
        if self.in_stack {
            // SAFETY: data is valid.
            unsafe {
                ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                    self.stack_ptr() as *mut T,
                    self.len,
                ));
            }
        } else {
            // SAFETY: data is valid.
            unsafe {
                ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.as_mut_ptr(), self.len));
                if !T::IS_ZST {
                    self.dealloc();
                }
            }
        }
    }
}

impl<T, const N: usize> FastVecData<T, N> {
    /// dealloc old memory
    ///
    /// # Safety
    /// - `self.ptr` points to heap memory. (self.in_stack == false)
    /// - `T` is not ZST.
    /// - `self.cap` is not zero. (zero should be stored on the stack.)
    /// - `self.cap` is old capacity.
    /// - Resources are transferred or released normally.
    #[inline]
    unsafe fn dealloc(&mut self) {
        debug_assert!(
            !T::IS_ZST && { self.cap > 0 },
            "Cannot dealloc zero sized memory."
        );

        // SAFETY: see function doc
        unsafe {
            dealloc(
                self.as_mut_ptr() as *mut _,
                Layout::from_size_align_unchecked(
                    mem::size_of::<T>() * self.cap,
                    mem::align_of::<T>(),
                ),
            );
        }
    }

    /// # Safety
    /// `self.cap` is not zero. (zero should be stored on the stack.)
    /// Resources are transferred or released normally.
    #[inline]
    pub(crate) unsafe fn try_dealloc(&mut self) {
        if !T::IS_ZST && !self.in_stack {
            unsafe {
                self.dealloc();
            }
        }
    }

    #[inline(always)]
    const fn stack_ptr(&self) -> *const T {
        &self.cache as *const [MaybeUninit<T>] as *const T
    }

    /// Refresh the ptr to ensure its validity.
    ///
    /// This will be automatically called by [`FastVec`](crate::FastVec),
    /// and users usually do not need to use it.
    ///
    /// For zero size types, this function has no overhead and the compiler can eliminate dead code.
    ///
    /// Currently, we use [`Cell`] to achieve internal variability, so this is not multi-threaded safe.
    ///
    /// But don't worry, this function is usually called by [`FastVec`](crate::FastVec),
    /// it is not [`Sync`], will not call this function in multi-threaded env.
    ///
    /// And the [`FastVecData`] reference generated by [`FastVec`](crate::FastVec) is
    /// already the pointer correct, which can be safely passed across threads.
    ///
    /// # Safety
    /// - Single threaded safety.
    /// - Multi-threaded ?
    #[inline(always)]
    pub unsafe fn refresh(&self) {
        if !T::IS_ZST && self.in_stack {
            self.ptr.set(self.stack_ptr() as *mut T);
        }
    }

    /// Create an empty [`FastVecData`].
    ///
    /// # Safety
    /// - Using [`FastVec`](crate::FastVec) to wrap the returned [`FastVecData`],
    /// - or manually call [`FastVecData::refresh`] before any method call.
    #[inline]
    pub(crate) const unsafe fn new() -> Self {
        unsafe {
            Self {
                cache: MaybeUninit::<[MaybeUninit<T>; N]>::uninit().assume_init(),
                ptr: Cell::new(ptr::dangling_mut::<T>()),
                len: 0,
                cap: N,
                in_stack: true,
            }
        }
    }

    /// Create an empty [`FastVecData`] with the specified capacity.
    ///
    /// # Safety
    /// - Using [`FastVec`](crate::FastVec) to wrap the returned [`FastVecData`],
    /// - or manually call [`FastVecData::refresh`] before any method call.
    #[inline]
    pub(crate) unsafe fn with_capacity(capacity: usize) -> Self {
        unsafe {
            let mut vec = Self::new();
            if capacity > N {
                vec.cap = capacity;
                vec.in_stack = false;
                if !T::IS_ZST {
                    let layout = Layout::array::<T>(capacity).unwrap();
                    vec.ptr.set(alloc(layout) as *mut T);
                }
            }
            vec
        }
    }

    /// # Safety
    /// - if T is not zero sized type, capacity > 0.
    /// - Using [`FastVec`](crate::FastVec) to wrap the returned [`FastVecData`],
    /// - or manually call [`FastVecData::refresh`] before any method call.
    /// - See [`Vec::from_raw_parts`]
    #[inline]
    pub(crate) const unsafe fn from_raw_parts(ptr: *mut T, length: usize, capacity: usize) -> Self {
        debug_assert!(
            { capacity > 0 } || { size_of::<T>() == 0 },
            "heap size of 0 is not allowed unless it is ZST.",
        );
        unsafe {
            Self {
                cache: MaybeUninit::<[MaybeUninit<T>; N]>::uninit().assume_init(),
                ptr: Cell::new(ptr),
                len: length,
                cap: capacity,
                in_stack: false,
            }
        }
    }

    /// Increase capacity
    ///
    /// # Safety
    /// - new_capacity * size_of::<T> <= isize::MAX
    /// - new_capacity > old_capacity
    #[inline(never)]
    unsafe fn grow(&mut self, new_capacity: usize) {
        debug_assert!(
            self.cap < new_capacity,
            "grow's new_capacity should be > old_capacity"
        );

        if T::IS_ZST {
            if new_capacity > N {
                self.in_stack = false;
                self.cap = new_capacity;
            } else {
                self.in_stack = true;
                self.cap = N;
            }
            return;
        }

        let old_ptr = self.as_mut_ptr();

        unsafe {
            if new_capacity > N {
                // SAFETY: new_capacity * size_of::<T> <= isize::MAX
                let new_layout = Layout::array::<T>(new_capacity).unwrap();
                let new_ptr = alloc(new_layout) as *mut T;

                // SAFETY: start <= end <= self.len
                ptr::copy_nonoverlapping(old_ptr, new_ptr, self.len);

                if !self.in_stack {
                    self.dealloc();
                }
                self.cap = new_capacity;
                self.in_stack = false;
                self.ptr.set(new_ptr);
            } else {
                crate::utils::cold_path();

                let new_ptr = self.stack_ptr() as *mut T;
                // SAFETY: start <= end <= self.len, stack data growth to stack is impossible.
                ptr::copy_nonoverlapping(old_ptr, new_ptr, self.len);

                // The data here cannot be on the stack.
                // in_stack -> self.cap == N, but new_capacity <= N
                self.dealloc();

                self.cap = N;
                self.in_stack = true;
                self.ptr.set(new_ptr);
            }
        }
    }

    /// Increase capacity
    ///
    /// # Safety
    /// - new_capacity * size_of::<T> <= isize::MAX
    /// - new_capacity > old_capacity
    #[inline(always)]
    unsafe fn grow_compare(&mut self, new_capacity: usize) {
        let new_capacity = core::cmp::max(new_capacity, self.cap << 1);
        unsafe {
            self.grow(new_capacity);
        }
    }

    /// Increase capacity, free up some space for inserting data.
    ///
    /// # Safety
    /// - new_capacity * size_of::<T> <= isize::MAX
    /// - new_capacity > old_capacity
    /// - new_capacity >= new_length (len + end - start)
    /// - start <= end
    /// - The vacant space needs to be inserted.
    /// - self.len need to be set.
    #[inline(never)]
    unsafe fn grow_split(&mut self, new_capacity: usize, start: usize, end: usize) {
        debug_assert!(
            self.cap < new_capacity,
            "grow's new_capacity should be > old_capacity"
        );
        debug_assert!(start <= end);
        debug_assert!(self.len + end - start <= new_capacity);

        if T::IS_ZST {
            if new_capacity > N {
                self.in_stack = false;
                self.cap = new_capacity;
            } else {
                self.in_stack = true;
                self.cap = N;
            }
            return;
        }

        let tail_len = self.len - start;
        let old_ptr = self.as_mut_ptr();

        unsafe {
            if new_capacity > N {
                // SAFETY: new_capacity * size_of::<T> <= isize::MAX
                let new_layout = Layout::array::<T>(new_capacity).unwrap();
                let new_ptr = alloc(new_layout) as *mut T;
                // SAFETY: start <= end <= self.len
                ptr::copy_nonoverlapping(old_ptr, new_ptr, start);
                ptr::copy_nonoverlapping(old_ptr.add(start), new_ptr.add(end), tail_len);

                if !self.in_stack {
                    self.dealloc();
                }
                self.in_stack = false;
                self.cap = new_capacity;
                self.ptr.set(new_ptr);
            } else {
                crate::utils::cold_path();

                let new_ptr = self.stack_ptr() as *mut T;

                // SAFETY: start <= end <= self.len, stack growth to stack is impossible.
                ptr::copy_nonoverlapping(old_ptr, new_ptr, start);
                ptr::copy_nonoverlapping(old_ptr.add(start), new_ptr.add(end), tail_len);

                // The data here cannot be on the stack.
                // in_stack -> self.cap == N, but new_capacity <= N
                self.dealloc();

                self.ptr.set(new_ptr);
                self.in_stack = true;
                self.cap = N;
            }
        }
    }

    /// Reduce capacity.
    ///
    /// # Safety
    /// - new_capacity < old_capacity
    /// - new_capacity >= self.len
    /// - **self.in_stack is false**
    #[inline(never)]
    unsafe fn reduce(&mut self, new_capacity: usize) {
        debug_assert!(self.len <= new_capacity && new_capacity < self.cap);
        debug_assert!(!self.in_stack);

        if T::IS_ZST {
            self.cap = if new_capacity <= N {
                self.in_stack = true;
                N
            } else {
                new_capacity
            };
            return;
        }

        // SAFETY:
        // - new_capacity < old_capacity
        // - new_capacity >= self.len
        // - self.in_stack is false
        // - T is not ZST
        unsafe {
            let old_ptr = self.as_mut_ptr();
            if new_capacity <= N {
                let new_ptr = self.stack_ptr() as *mut T;
                ptr::copy_nonoverlapping(old_ptr, new_ptr, self.len);

                self.dealloc();

                self.cap = N;
                self.in_stack = true;
                self.ptr.set(new_ptr);
            } else {
                let new_layout = Layout::array::<T>(new_capacity).unwrap();
                let new_ptr = alloc(new_layout) as *mut T;
                ptr::copy_nonoverlapping(old_ptr, new_ptr, self.len);

                self.dealloc();

                self.cap = new_capacity;
                self.ptr.set(new_ptr);
            };
        }
    }

    /// Returns a raw pointer to the vector’s buffer, or a dangling raw pointer
    /// valid for zero sized reads if the vector didn’t allocate.
    #[inline(always)]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr.get()
    }

    /// Returns a raw mutable pointer to the vector’s buffer, or a dangling raw pointer
    /// valid for zero sized reads if the vector didn’t allocate.
    #[inline(always)]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.get()
    }

    /// Returns the total number of elements the vector can hold without reallocating.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut state: FastVec<i32, 8> = [1, 2, 3, 4].into();
    /// let vec = state.data();
    /// assert_eq!(vec.capacity(), 8);
    /// assert_eq!(vec.len(), 4);
    ///
    /// vec.extend([1, 2, 3,  4, 5]);
    /// assert!(vec.capacity() >= 9);
    /// assert_eq!(vec.len(), 9);
    /// ```
    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        self.cap
    }

    /// Returns the number of elements in the vector, also referred to as its length.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut state: FastVec<i32, 8> = [1, 2, 3, 4].into();
    /// let vec = state.data();
    /// assert_eq!(vec.capacity(), 8);
    /// assert_eq!(vec.len(), 4);
    ///
    /// vec.extend([1, 2, 3,  4, 5]);
    /// assert!(vec.capacity() >= 9);
    /// assert_eq!(vec.len(), 9);
    /// ```
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut state: FastVec<i32> = FastVec::new();
    /// let vec = state.data();
    /// assert!(vec.is_empty());
    ///
    /// vec.push(1);
    /// assert!(!vec.is_empty());
    /// ```
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns `true` if the data is stored in stack.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut state: FastVec<i32, 8> = [1, 2, 3, 4].into();
    /// let vec = state.data();
    /// assert!(vec.in_stack());
    ///
    /// vec.extend([1, 2, 3,  4, 5]);
    /// assert!(!vec.in_stack());
    /// ```
    #[inline(always)]
    pub const fn in_stack(&self) -> bool {
        self.in_stack
    }

    /// Reserves the minimum capacity for at least `additional` more elements to be inserted in the given vector.
    ///
    /// If the existing capacity is sufficient, this will not do anything.
    ///
    /// See [`Vec::reserve`] .
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut state: FastVec<i32, 8> = FastVec::new();
    /// let vec = state.data();
    ///
    /// vec.reserve(5); // do nothing
    /// assert!(vec.in_stack());
    /// assert_eq!(vec.capacity(), 8);
    ///
    /// vec.reserve(10);
    /// assert!(!vec.in_stack());
    /// assert!(vec.capacity() >= 10);
    ///
    /// vec.reserve(6); // do nothing
    /// assert!(!vec.in_stack());
    /// assert!(vec.capacity() >= 10);
    /// ```
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        let new_capacity = self.len + additional;
        if new_capacity > self.cap {
            // SAFETY: new_capacity > self.cap
            unsafe {
                self.grow_compare(new_capacity);
            }
        }
    }

    /// Reserves the minimum capacity for at least `additional` more elements to be inserted in the given vector.
    ///
    /// If the existing capacity is sufficient, this will not do anything.
    ///
    /// If it is possible to move from heap to stack, stack space will be used,
    /// and the target capacity is not precise at this time.
    ///
    /// See [`Vec::reserve_exact`] .
    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        let new_capacity = self.len + additional;
        if new_capacity > self.cap {
            // SAFETY: new_capacity > self.cap
            unsafe {
                self.grow(new_capacity);
            }
        }
    }

    /// Shrinks the capacity of the vector with a lower bound.
    ///
    /// The capacity will remain at least as large as both the length and the supplied value.
    ///
    /// This function may move data from the heap to the stack.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut state: FastVec<i32, 10> = [1, 2, 3].into();
    /// let vec = state.data();
    ///
    /// assert!(vec.capacity() == 10);
    /// vec.shrink_to(4);
    /// assert!(vec.capacity() == 10);
    ///
    /// vec.reserve(15);
    /// assert!(vec.capacity() >= 15);
    /// assert!(!vec.in_stack());
    ///
    /// vec.shrink_to(4);
    /// assert!(vec.capacity() == 10);
    /// assert!(vec.in_stack());
    /// ```
    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        if self.in_stack {
            return;
        }
        let capacity = min_capacity.min(self.cap).max(self.len);
        if capacity != self.cap {
            // SAFETY: new_capacity >= self.len, new_capacity < old_capacity, !in_stack
            unsafe {
                self.reduce(capacity);
            }
        }
    }

    /// Shrinks the capacity of the vector with a lower bound.
    ///
    /// The capacity will remain at least as large as both the length and the supplied value.
    ///
    /// This function may move data from the heap to the stack.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut state: FastVec<i32, 10> = [1, 2, 3].into();
    /// let vec = state.data();
    ///
    /// assert!(vec.capacity() == 10);
    /// vec.shrink_to_fit();
    /// assert!(vec.capacity() == 10);
    ///
    /// vec.reserve(15);
    /// assert!(vec.capacity() >= 15);
    /// assert!(!vec.in_stack());
    ///
    /// vec.shrink_to_fit();
    /// assert!(vec.capacity() == 10);
    /// assert!(vec.in_stack());
    /// ```
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        if self.in_stack {
            return;
        }
        if self.cap != self.len {
            // SAFETY: new_capacity >= self.len, new_capacity < old_capacity, !in_stack
            unsafe {
                self.reduce(self.len);
            }
        }
    }

    /// Converts the vector into [`Vec<T>`](Vec).
    ///
    /// - If the data is in the stack, then the created [`Vec`]'s capacity is accurate.
    /// - If the data is in the heap, then transfer the pointer directly to ensure maximum efficiency.
    #[inline]
    pub(crate) fn into_vec(mut self) -> Vec<T> {
        if self.in_stack {
            let len = self.len;
            let mut vec: Vec<T> = Vec::with_capacity(len);
            unsafe {
                ptr::copy_nonoverlapping(self.stack_ptr(), vec.as_mut_ptr(), len);
                vec.set_len(len);
            }
            self.len = 0;
            vec
        } else {
            let vec = unsafe { Vec::from_raw_parts(self.as_mut_ptr(), self.len, self.cap) };
            self.len = 0;
            self.in_stack = true;
            vec
        }
    }

    /// Converts the vector into [`Vec<T>`](Vec).
    ///
    /// - If the data is in the stack, then the created [`Vec`]'s capacity is accurate.
    /// - If the data is in the heap, then transfer the pointer and call [`Vec::shrink_to_fit`] .
    #[inline]
    pub(crate) fn shrink_into_vec(self) -> Vec<T> {
        let mut vec = self.into_vec();
        vec.shrink_to_fit();
        vec
    }

    /// Converts the vector into [`Box<[T]>`](Box).
    #[inline]
    pub(crate) fn into_boxed_slice(self) -> Box<[T]> {
        self.into_vec().into_boxed_slice()
    }

    /// Shortens the vector, keeping the first `len` elements and dropping the rest.
    ///
    /// Note that this method has no effect on the allocated capacity of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut state: FastVec<i32, 3> = [1, 2, 3, 4].into();
    /// let vec = state.data();
    ///
    /// assert!(!vec.in_stack());
    ///
    /// vec.truncate(2);
    ///
    /// assert_eq!(vec, &[1, 2]);
    /// assert!(!vec.in_stack());
    /// ```
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        if len < self.len {
            unsafe {
                ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                    self.as_mut_ptr().add(len),
                    self.len - len,
                ));
                self.len = len;
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
    /// # use fastvec::FastVec;
    /// let mut state: FastVec<i32, 3> = [1, 2, 3, 4].into();
    /// let vec = state.data();
    ///
    /// vec.clear();
    ///
    /// assert_eq!(vec, &[]);
    /// assert!(!vec.in_stack());
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        if self.len > 0 {
            unsafe {
                ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.as_mut_ptr(), self.len));
                self.len = 0;
            }
        }
    }

    /// Extracts a slice containing the entire vector.
    #[inline(always)]
    pub const fn as_slice(&self) -> &[T] {
        // SAFETY: self.ptr is refreshed, so data is valid.
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len) }
    }

    /// Extracts a mutable slice of the entire vector.
    #[inline(always)]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: self.ptr is refreshed, so data is valid.
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len) }
    }

    /// Forces the length of the vector to `new_len`.
    ///
    /// # Safety
    /// - `new_len` must be less than or equal to `capacity()`.
    /// - If `new_len > old_len`, the elements at `old_len..new_len` must be initialized.
    /// - If `new_len < old_len`, the elements as `new_len..old_len` must be dropped.
    #[inline(always)]
    pub const unsafe fn set_len(&mut self, new_len: usize) {
        self.len = new_len
    }

    /// Removes an element from the vector and returns it.
    ///
    /// The removed element is replaced by the last element of the vector, so it's O(1) time complexity.
    ///
    /// Note that this method has no effect on the allocated capacity of the vector.
    ///
    /// # Panics
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut state: FastVec<&'static str> = ["foo", "bar", "baz", "qux"].into();
    /// let v = state.data();
    ///
    /// assert_eq!(v.swap_remove(1), "bar");
    /// assert_eq!(v, &["foo", "qux", "baz"]);
    /// assert_eq!(v.swap_remove(0), "foo");
    /// assert_eq!(v, &["baz", "qux"]);
    /// ```
    #[inline]
    pub const fn swap_remove(&mut self, index: usize) -> T {
        assert!(index < self.len, "removal index should be < len");
        // SAFETY: self.ptr is refreshed, so data is valid.
        unsafe {
            // We replace self[index] with the last element.
            let value = ptr::read(self.as_ptr().add(index));
            if !T::IS_ZST {
                let base_ptr = self.as_mut_ptr();
                ptr::copy(base_ptr.add(self.len - 1), base_ptr.add(index), 1);
            }
            self.len -= 1;
            value
        }
    }

    /// Inserts an element at position `index` within the vector, shifting all elements after it to the right.
    ///
    /// If the cache(stack) capacity is insufficient, the data will be moved to the heap.
    ///
    /// # Time complexity
    /// O(N)
    ///
    /// # Panics
    /// - Panics if `index > len` .
    /// - Panics if the new capacity exceeds `isize::MAX` bytes.
    ///
    /// # Panics
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut state: FastVec<char> = ['a', 'b', 'c'].into();
    /// let vec = state.data();
    ///
    /// vec.insert(1, 'd');
    /// assert_eq!(vec, &['a', 'd', 'b', 'c']);
    ///
    /// vec.insert(4, 'e');
    /// assert_eq!(vec, &['a', 'd', 'b', 'c', 'e']);
    /// ```
    pub fn insert(&mut self, index: usize, element: T) {
        let len = self.len;

        assert!(index <= self.len, "inserted index should be <= len");

        if len == self.cap {
            cold_path();

            // SAFETY:
            // - self.ptr is refreshed, so data is valid.
            // - capacity is sufficent, index <= self.len.
            unsafe {
                self.grow_split(
                    if self.cap > 0 {
                        self.cap << 1
                    } else {
                        crate::utils::min_cap::<T>()
                    },
                    index,
                    index + 1,
                );
                if T::IS_ZST {
                    core::mem::forget(element);
                } else {
                    ptr::write(self.as_mut_ptr(), element);
                }
                self.len = len + 1;
            }
        } else {
            // SAFETY: self.ptr is refreshed, so data is valid.
            unsafe {
                if T::IS_ZST {
                    core::mem::forget(element);
                } else {
                    let index_ptr = self.as_mut_ptr().add(index);
                    core::hint::assert_unchecked(!index_ptr.is_null() && index_ptr.is_aligned());
                    ptr::copy(index_ptr, index_ptr.add(1), len - index);
                    ptr::write(index_ptr, element);
                }
                self.len = len + 1;
            }
        }
    }

    /// Removes and returns the element at position index within the vector, shifting all elements after it to the left.
    ///
    /// Note that this method has no effect on the allocated capacity of the vector.
    ///
    /// # Time complexity
    /// O(N)
    ///
    /// # Panics
    /// - Panics if `index >= len` .
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut state: FastVec<char> = ['a', 'b', 'c'].into();
    /// let v = state.data();
    ///
    /// assert_eq!(v.remove(1), 'b');
    /// assert_eq!(v, &['a', 'c']);
    /// ```
    pub const fn remove(&mut self, index: usize) -> T {
        assert!(index < self.len, "removal index should be < len");
        // SAFETY: self.ptr is refreshed, so data is valid.
        unsafe {
            // We replace self[index] with the last element.
            let value = ptr::read(self.as_ptr().add(index));
            if !T::IS_ZST {
                let base_ptr = self.as_mut_ptr();
                let index_1 = index + 1;
                ptr::copy(
                    base_ptr.add(index_1),
                    base_ptr.add(index),
                    self.len - index_1,
                );
            }
            self.len -= 1;
            value
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// Note that this method has no effect on the allocated capacity of the vector.
    ///
    /// # Time complexity
    /// O(N)
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut state: FastVec<_> = [1, 2, 3, 4, 5].into();
    /// let vec = state.data();
    ///
    /// let keep = [false, true, true, false, true];
    /// let mut iter = keep.iter();
    /// vec.retain(|_| *iter.next().unwrap());
    /// assert_eq!(vec, &[2, 3, 5]);
    /// ```
    #[inline]
    pub fn retain<F: FnMut(&T) -> bool>(&mut self, mut f: F) {
        self.retain_mut(|v| f(v));
    }

    /// Retains only the elements specified by the predicate, passing a mutable reference to it.
    ///
    /// Note that this method has no effect on the allocated capacity of the vector.
    ///
    /// # Time complexity
    /// O(N)
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut state: FastVec<_> = [1, 2, 3, 4, 5].into();
    /// let vec = state.data();
    ///
    /// let keep = [false, true, true, false, true];
    /// let mut iter = keep.iter();
    /// vec.retain_mut(|x| { *x += 10; *iter.next().unwrap() });
    /// assert_eq!(vec, &[12, 13, 15]);
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

    /// Removes all but the first of consecutive elements in the vector satisfying a given equality relation.
    ///
    /// Note that this method has no effect on the allocated capacity of the vector.
    ///
    /// # Time complexity
    /// O(N)
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<_> = ["foo", "bar", "Bar", "baz", "bar"].into();
    ///
    /// vec.data().dedup_by(|a, b| a.eq_ignore_ascii_case(b));
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

    /// Removes all but the first of consecutive elements in the vector that resolve to the same key.
    ///
    /// Note that this method has no effect on the allocated capacity of the vector.
    ///
    /// # Time Complexity
    /// O(N)
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<_> = [10, 20, 21, 30, 20].into();
    ///
    /// vec.data().dedup_by_key(|i| *i / 10);
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

    /// Appends an element to the back of a collection.
    ///
    /// If the cache(stack) capacity is insufficient, the data will be moved to the heap.
    ///
    /// # Time complexity
    /// O(1)
    ///
    /// # Panics
    /// - Panics if the new capacity exceeds `isize::MAX` bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<_> = [1, 2].into();
    ///
    /// vec.data().push(3);
    ///
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    #[inline]
    pub fn push(&mut self, value: T) {
        let len = self.len;
        if len == self.cap {
            cold_path();
            unsafe {
                self.grow(if self.cap > 0 {
                    self.cap << 1
                } else {
                    crate::utils::min_cap::<T>()
                });
            }
        }

        unsafe {
            ptr::write(self.as_mut_ptr().add(len), value);
        }
        self.len = len + 1;
    }

    /// Appends an element to the back of a collection without checking capacity.
    ///
    /// # Safety
    /// len < capacity (before push).
    #[inline(always)]
    pub unsafe fn push_unchecked(&mut self, value: T) {
        let len = self.len;
        unsafe { ptr::write(self.as_mut_ptr().add(len), value) }
        self.len = len + 1;
    }

    /// Removes the last element from a vector and returns it, or `None` if it is empty.
    ///
    /// Note that this method has no effect on the allocated capacity of the vector.
    ///
    /// # Time complexity
    /// O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<_> = [1, 2, 3].into();
    /// assert_eq!(vec.data().pop(), Some(3));
    /// assert_eq!(vec, [1, 2]);
    /// ```
    #[inline]
    pub const fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            cold_path();
            None
        } else {
            unsafe {
                self.len -= 1;
                core::hint::assert_unchecked(self.len < self.capacity());
                Some(ptr::read(self.as_mut_ptr().add(self.len)))
            }
        }
    }

    /// Removes and returns the last element from a vector if the predicate returns `true`,
    /// or `None` if the predicate returns `false` or the vector is empty (the predicate will not be called in that case).
    ///
    /// Note that this method has no effect on the allocated capacity of the vector.
    ///
    /// # Time complexity
    /// O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<_> = [1, 2, 3, 4].into();
    /// let pred = |x: &mut i32| *x % 2 == 0;
    ///
    /// assert_eq!(vec.data().pop_if(pred), Some(4));
    /// assert_eq!(vec, [1, 2, 3]);
    /// assert_eq!(vec.data().pop_if(pred), None);
    /// ```
    #[inline]
    pub fn pop_if(&mut self, predicate: impl FnOnce(&mut T) -> bool) -> Option<T> {
        if self.len == 0 {
            return None;
        }

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

    /// Removes the subslice indicated by the given range from the vector,
    /// returning a double-ended iterator over the removed subslice.
    ///
    /// If the cache(stack) capacity is insufficient, the data will be moved to the heap.
    ///
    /// Note that this method has no effect on the allocated capacity of the **other vector**.
    ///
    /// # Time complexity
    /// O(N)
    ///
    /// # Panics
    /// - Panics if the new capacity exceeds `isize::MAX` bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<_, 5> = [1, 2, 3].into();
    /// let mut vec2: FastVec<_, 3> = [4, 5, 6].into();
    /// vec.data().append(vec2.data());
    ///
    /// assert_eq!(vec, [1, 2, 3, 4, 5, 6]);
    /// assert_eq!(vec2, []);
    ///
    /// assert!(!vec.in_stack());
    /// ```
    #[inline]
    pub fn append<const P: usize>(&mut self, other: &mut FastVecData<T, P>) {
        let new_len = self.len + other.len;
        if new_len > self.cap {
            unsafe {
                self.grow_compare(new_len);
            }
        }
        if !T::IS_ZST {
            unsafe {
                ptr::copy_nonoverlapping(
                    other.as_ptr(),
                    self.as_mut_ptr().add(self.len),
                    other.len,
                );
            }
        }
        other.len = 0;
        self.len = new_len;
    }

    /// Splits the collection into two at the given index.
    ///
    /// # Time complexity
    /// O(N)
    ///
    /// # Panics
    /// Panics if `at > len`.
    ///
    /// # Safety
    /// - Using [`FastVec`](crate::FastVec) to host the returned [`FastVecData`],
    /// - or manually call [`FastVecData::refresh`] before any method call.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<_> = ['a', 'b', 'c'].into();
    /// let vec2: FastVec<_> = vec.data().split_off(1);
    ///
    /// assert_eq!(vec, ['a']);
    /// assert_eq!(vec2, ['b', 'c']);
    /// ```
    #[inline]
    pub fn split_off(&mut self, at: usize) -> crate::fast_vec::FastVec<T, N> {
        assert!(at <= self.len, "the `at` of split off should be <= len");
        let other_len = self.len - at;

        unsafe {
            let mut state = <crate::fast_vec::FastVec<T, N>>::with_capacity(other_len);
            let other = state.data();
            other.len = other_len;
            self.len = at;
            if !T::IS_ZST {
                ptr::copy_nonoverlapping(self.as_ptr().add(at), other.as_mut_ptr(), other_len);
            }
            state
        }
    }

    /// Resizes the vector in-place so that `len` is equal to `new_len`.
    ///
    /// If the cache(stack) capacity is insufficient, the data will be moved to the heap.
    /// If the capacity is sufficient, it will not affect the allocated memory.
    ///
    /// # Panics
    /// - Panics if the new capacity exceeds `isize::MAX` bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<_> = [1, 2, 3].into();
    /// vec.data().resize_with(5, Default::default);
    ///
    /// let mut vec: FastVec<i32> = [].into();
    /// let mut p = 1;
    /// vec.data().resize_with(4, || { p *= 2; p });
    /// assert_eq!(vec, [2, 4, 8, 16]);
    /// ```
    pub fn resize_with<F: FnMut() -> T>(&mut self, new_len: usize, mut f: F) {
        if new_len > self.cap {
            unsafe {
                self.grow_compare(new_len);
            }
        }

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

    /// Consumes and leaks the vector, returning a mutable reference to the contents, `&'a mut [T]`.
    ///
    /// This will first move the data to the heap to ensure that the returned references are valid.
    #[inline]
    pub(crate) fn leak<'a>(self) -> &'a mut [T] {
        self.into_vec().leak()
    }

    /// Returns the remaining spare capacity of the vector as a slice of `MaybeUninit<T>`.
    ///
    /// The returned slice can be used to fill the vector with data (e.g. by reading from a file)
    /// before marking the data as initialized using the [`set_len`](FastVecData::set_len) method.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<i32> = FastVec::with_capacity(10);
    /// let v = vec.data();
    ///
    /// let uninit = v.spare_capacity_mut();
    /// uninit[0].write(0);
    /// uninit[1].write(1);
    /// uninit[2].write(2);
    ///
    /// unsafe {
    ///     v.set_len(3);
    /// }
    ///
    /// assert_eq!(v, &[0, 1, 2]);
    /// ```
    #[inline]
    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        unsafe {
            slice::from_raw_parts_mut(
                self.as_mut_ptr().add(self.len) as *mut MaybeUninit<T>,
                self.cap - self.len,
            )
        }
    }
}

impl<T: Clone, const N: usize> FastVecData<T, N> {
    /// Resizes the vector in-place so that `len` is equal to `new_len`.
    ///
    /// If the cache(stack) capacity is insufficient, the data will be moved to the heap.
    /// If the capacity is sufficient, it will not affect the allocated memory.
    ///
    /// # Panics
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<_> = ["hello"].into();
    /// vec.data().resize(3, "world");;
    /// assert_eq!(vec, ["hello", "world", "world"]);
    ///
    /// let mut vec: FastVec<_> = ['a', 'b', 'c', 'd'].into();
    /// vec.data().resize(2, '_');
    /// assert_eq!(vec, ['a', 'b']);
    /// ```
    pub fn resize(&mut self, new_len: usize, value: T) {
        if new_len > self.cap {
            unsafe { self.grow(new_len) };
        }

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

    /// Clones and appends all elements in a slice to the vector.
    ///
    /// If the cache(stack) capacity is insufficient, the data will be moved to the heap.
    ///
    /// This is similar to `Extend` trait, but it is faster because we know the number of insertions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<_> = [1].into();
    /// vec.data().extend_from_slice(&[2, 3, 4]);
    /// assert_eq!(vec, [1, 2, 3, 4]);
    /// ```
    pub fn extend_from_slice(&mut self, other: &[T]) {
        let new_len = other.len() + self.len;
        if new_len > self.cap {
            unsafe {
                self.grow_compare(new_len);
            }
        }
        for item in other {
            unsafe {
                self.push_unchecked(item.clone());
            }
        }
    }

    /// Clones elements from the given range within the vector and appends them to the end.
    ///
    /// The range `src` must form a valid subslice of the vector.
    ///
    /// # Panics
    /// - Starting index is greater than the end index.
    /// - The index is greater than the length of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<_> = ['a', 'b', 'c', 'd', 'e'].into();
    /// vec.data().extend_from_within(2..);
    /// assert_eq!(vec, ['a', 'b', 'c', 'd', 'e', 'c', 'd', 'e']);
    /// ```
    pub fn extend_from_within<R: core::ops::RangeBounds<usize>>(&mut self, src: R) {
        let (start, end) = crate::utils::split_range_bound(&src, self.len);
        let new_len = end - start + self.len;
        if new_len > self.cap {
            unsafe {
                self.grow_compare(new_len);
            }
        }

        let base_ptr = self.as_mut_ptr();
        for index in start..end {
            unsafe {
                self.push_unchecked((&*base_ptr.add(index)).clone());
            }
        }
    }

    /// Clone self to a new [`FastVec<T, N>`](crate::FastVec) .
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<_> = [1, 2, 3, 4].into();
    ///
    /// let vec2 = vec.data().clone_up();
    /// assert_eq!(vec, vec2);
    /// assert_eq!(vec, [1, 2, 3, 4]);
    /// ```
    pub fn clone_up(&self) -> crate::fast_vec::FastVec<T, N> {
        let mut vec = <crate::fast_vec::FastVec<T, N>>::with_capacity(self.len);
        let dst = vec.data();
        for item in self.as_slice() {
            unsafe {
                dst.push_unchecked(item.clone());
            }
        }
        vec
    }

    /// Clone from [`FastVec<T, N>`](crate::FastVec)  .
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<_> = [1, 2, 3, 4].into();
    ///
    /// let mut vec2 = <FastVec<i32>>::new();
    ///
    /// vec2.data().clone_from(vec.data());
    ///
    /// assert_eq!(vec, vec2);
    /// assert_eq!(vec2, [1, 2, 3, 4]);
    /// ```
    pub fn clone_from<const P: usize>(&mut self, source: &FastVecData<T, P>) {
        self.clear();
        self.reserve(source.len);
        for item in source.as_slice() {
            unsafe {
                self.push_unchecked(item.clone());
            }
        }
    }
}

impl<T, const N: usize, const P: usize> FastVecData<[T; P], N> {
    /// Takes a `FastVecData<[T; N]>` and flattens it into a `FastVecData<T, S>`.
    ///
    /// If the data is on the stack, it needs to be copied.
    /// If the data is on the heap, it only requires transferring the pointer.
    pub(crate) fn into_flattened<const S: usize>(mut self) -> FastVecData<T, S> {
        let new_len = self.len * P;
        unsafe {
            if self.in_stack || P == 0 {
                let mut vec = <FastVecData<T, S>>::with_capacity(new_len);
                ptr::copy_nonoverlapping(self.as_ptr() as *const T, vec.as_mut_ptr(), self.len * P);
                vec.len = new_len;
                self.len = 0;

                vec
            } else {
                debug_assert!(
                    T::IS_ZST || self.cap * P > 0,
                    "heap size of 0 is not allowed unless it is ZST."
                );
                let vec = <FastVecData<T, S>>::from_raw_parts(
                    self.as_mut_ptr() as *mut T,
                    new_len,
                    self.cap * P,
                );
                self.len = 0;
                self.in_stack = true;
                vec
            }
        }
    }
}

impl<T: PartialEq, const N: usize> FastVecData<T, N> {
    /// Removes consecutive repeated elements in the vector according to the `PartialEq` trait implementation.
    ///
    /// If the vector is sorted, this removes all duplicates.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<i32> = [1, 2, 2, 3, 3, 2].into();
    ///
    /// vec.data().dedup();
    ///
    /// assert_eq!(vec, [1, 2, 3, 2]);
    /// ```
    #[inline]
    pub fn dedup(&mut self) {
        self.dedup_by(|x, y| PartialEq::eq(x, y));
    }
}

impl<'a, T: 'a + Clone, const N: usize> Extend<&'a T> for FastVecData<T, N> {
    /// Clone values from iterators.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<i32> = [1, 2, 3].into();
    ///
    /// vec.data().extend(&[4, 5, 6]);
    ///
    /// assert_eq!(vec, [1, 2, 3, 4, 5, 6]);
    /// ```
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        for item in iter {
            self.push(item.clone());
        }
    }
}

impl<T, const N: usize> Extend<T> for FastVecData<T, N> {
    /// Extends a collection with the contents of an iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::FastVec;
    /// let mut vec: FastVec<i32> = [1, 2, 3].into();
    ///
    /// vec.data().extend([4, 5, 6]);
    ///
    /// assert_eq!(vec, [1, 2, 3, 4, 5, 6]);
    /// ```
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.push(item);
        }
    }
}

crate::utils::impl_commen_traits!(FastVecData<T, N>);

impl<T, U, const N: usize> PartialEq<FastVecData<U, N>> for FastVecData<T, N>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &FastVecData<U, N>) -> bool {
        PartialEq::eq(self.as_slice(), other.as_slice())
    }
}

impl<'a, T: Clone, const N: usize> From<&'a FastVecData<T, N>> for alloc::borrow::Cow<'a, [T]> {
    fn from(v: &'a FastVecData<T, N>) -> alloc::borrow::Cow<'a, [T]> {
        alloc::borrow::Cow::Borrowed(v.as_slice())
    }
}

/// An iterator that removes the items from a [`FastVec`](crate::FastVec) and yields them by value.
///
/// See [`FastVecData::drain`] .
pub struct Drain<'a, T: 'a, const N: usize> {
    tail_start: usize,
    tail_len: usize,
    iter: core::slice::Iter<'a, T>,
    vec: ptr::NonNull<FastVecData<T, N>>,
}

impl<T, const N: usize> FastVecData<T, N> {
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
    /// # use fastvec::{FastVec, fastvec};
    /// let mut v: FastVec<_> = fastvec![1, 2, 3];
    /// let u: Vec<_> = v.data().drain(1..).collect();
    /// assert_eq!(v, [1]);
    /// assert_eq!(u, [2, 3]);
    ///
    /// // A full range clears the vector, like `clear()` does
    /// v.data().drain(..);
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

        if T::IS_ZST {
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

/// A splicing iterator for [`FastVec`](crate::FastVec).
///
/// See [`FastVecData::splice`] .
#[derive(Debug)]
pub struct Splice<'a, I: ExactSizeIterator + 'a, const N: usize> {
    pub(super) drain: Drain<'a, I::Item, N>,
    pub(super) replace_with: I,
}

impl<T, const N: usize> FastVecData<T, N> {
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
    ///
    /// # Examples
    ///
    /// ```
    /// # use fastvec::{fastvec, FastVec};
    /// let mut v: FastVec<_> = fastvec![1, 2, 3, 4];
    /// let new = [7, 8, 9];
    /// let u: Vec<_> = v.data().splice(1..3, new).collect();
    /// assert_eq!(v, [1, 7, 8, 9, 4]);
    /// assert_eq!(u, [2, 3]);
    /// ```
    ///
    /// Using `splice` to insert new items into a vector efficiently at a specific position
    /// indicated by an empty range:
    ///
    /// ```
    /// # use fastvec::{fastvec, FastVec};
    /// let mut v: FastVec<_> = fastvec![1, 5];
    /// let new = [2, 3, 4];
    /// v.data().splice(1..1, new);
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
        self.drain.iter = [].iter();

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

            let need_capacity = new_tail_start + self.drain.tail_len;
            if vec.cap < need_capacity {
                vec.grow_compare(need_capacity);
            }

            if new_tail_start != self.drain.tail_start {
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
/// See [`Vec::extract_if`] .
pub struct ExtractIf<'a, T, F: FnMut(&mut T) -> bool, const N: usize> {
    vec: &'a mut FastVecData<T, N>,
    idx: usize,
    end: usize,
    del: usize,
    old_len: usize,
    pred: F,
}

impl<T, const N: usize> FastVecData<T, N> {
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
    /// # use fastvec::{fastvec, FastVec};
    /// let mut numbers: FastVec<_> = fastvec![1, 2, 3, 4, 5, 6, 8, 9, 11, 13, 14, 15];
    ///
    /// let evens = numbers.data().extract_if(.., |x| *x % 2 == 0).collect::<FastVec<_, 10>>();
    /// let odds = numbers;
    ///
    /// assert_eq!(evens, [2, 4, 6, 8, 14]);
    /// assert_eq!(odds, [1, 3, 5, 9, 11, 13, 15]);
    /// ```
    ///
    /// Using the range argument to only process a part of the vector:
    ///
    /// ```
    /// # use fastvec::{fastvec, FastVec};
    /// let mut items: FastVec<_> = fastvec![0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2];
    /// let ones = items.data().extract_if(7.., |x| *x == 1).collect::<Vec<_>>();
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
