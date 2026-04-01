//! Internal helpers shared by vector implementations.

/// Creates an initialized value for a zero-sized type.
///
/// # Safety
///
/// `T` must be a zero-sized type.
#[inline(always)]
pub(super) const unsafe fn zst_init<T>() -> T {
    debug_assert!(::core::mem::size_of::<T>() == 0);

    #[expect(clippy::uninit_assumed_init)]
    unsafe {
        core::mem::MaybeUninit::uninit().assume_init()
    }
}

#[cold]
#[inline(never)]
pub(super) const fn cold_path() {}

/// Compile-time flag for whether a type is zero-sized.
pub(super) trait IsZST {
    const IS_ZST: bool;
}

impl<T> IsZST for T {
    /// A flag used to indicate the ZST (zero sized type).
    ///
    /// This will be optimized by the compiler and will not take up additional space.
    ///
    /// Don't worry about the additional overhead of branching statements.
    const IS_ZST: bool = ::core::mem::size_of::<T>() == 0;
}

/// Chooses a minimum non-zero capacity heuristic for type `T`.
#[inline(always)]
pub(super) const fn min_cap<T>() -> usize {
    let size = ::core::mem::size_of::<T>();
    if size < 1 {
        8
    } else if size <= 1024 {
        4
    } else {
        1
    }
}

/// Converts generic range bounds into normalized `(start, end)` indices.
///
/// # Panics
///
/// Panics if `start > end` or `end > len`.
#[inline(never)]
pub(super) fn split_range_bound(
    src: &impl core::ops::RangeBounds<usize>,
    len: usize,
) -> (usize, usize) {
    let start = match src.start_bound() {
        core::ops::Bound::Included(&i) => i,
        core::ops::Bound::Excluded(&i) => i + 1,
        core::ops::Bound::Unbounded => 0,
    };

    let end = match src.end_bound() {
        core::ops::Bound::Included(&i) => i + 1,
        core::ops::Bound::Excluded(&i) => i,
        core::ops::Bound::Unbounded => len,
    };

    assert!(start <= end, "drain start greater than end");
    assert!(end <= len, "drain end out of bounds");
    (start, end)
}

macro_rules! impl_common_traits {
    ($name:ty) => {
        impl<T, const N: usize> core::ops::Deref for $name {
            type Target = [T];
            #[inline]
            fn deref(&self) -> &Self::Target {
                self.as_slice()
            }
        }

        impl<T, const N: usize> core::ops::DerefMut for $name {
            #[inline]
            fn deref_mut(&mut self) -> &mut Self::Target {
                self.as_mut_slice()
            }
        }

        impl<T: core::fmt::Debug, const N: usize> core::fmt::Debug for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                core::fmt::Debug::fmt(self.as_slice(), f)
            }
        }

        impl<T, const N: usize> core::convert::AsRef<[T]> for $name {
            #[inline]
            fn as_ref(&self) -> &[T] {
                self.as_slice()
            }
        }

        impl<T, const N: usize> core::convert::AsRef<$name> for $name {
            #[inline]
            fn as_ref(&self) -> &$name {
                self
            }
        }

        impl<T, const N: usize> core::convert::AsMut<[T]> for $name {
            #[inline]
            fn as_mut(&mut self) -> &mut [T] {
                self.as_mut_slice()
            }
        }

        impl<T, const N: usize> core::convert::AsMut<$name> for $name {
            #[inline]
            fn as_mut(&mut self) -> &mut $name {
                self
            }
        }

        impl<T, const N: usize> core::borrow::Borrow<[T]> for $name {
            #[inline]
            fn borrow(&self) -> &[T] {
                self.as_slice()
            }
        }

        impl<T, const N: usize> core::borrow::BorrowMut<[T]> for $name {
            #[inline]
            fn borrow_mut(&mut self) -> &mut [T] {
                self.as_mut_slice()
            }
        }

        impl<T: core::hash::Hash, const N: usize> core::hash::Hash for $name {
            #[inline]
            fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
                core::hash::Hash::hash(self.as_slice(), state);
            }
        }

        impl<T, I: core::slice::SliceIndex<[T]>, const N: usize> core::ops::Index<I> for $name {
            type Output = <I as core::slice::SliceIndex<[T]>>::Output;
            #[inline]
            fn index(&self, index: I) -> &Self::Output {
                core::ops::Index::index(self.as_slice(), index)
            }
        }

        impl<T, I: core::slice::SliceIndex<[T]>, const N: usize> core::ops::IndexMut<I> for $name {
            #[inline]
            fn index_mut(&mut self, index: I) -> &mut Self::Output {
                core::ops::IndexMut::index_mut(self.as_mut_slice(), index)
            }
        }

        impl<'a, T, const N: usize> IntoIterator for &'a $name {
            type Item = &'a T;
            type IntoIter = core::slice::Iter<'a, T>;
            #[inline]
            fn into_iter(self) -> Self::IntoIter {
                self.as_slice().iter()
            }
        }

        impl<'a, T, const N: usize> IntoIterator for &'a mut $name {
            type Item = &'a mut T;
            type IntoIter = core::slice::IterMut<'a, T>;
            #[inline]
            fn into_iter(self) -> Self::IntoIter {
                self.as_mut_slice().iter_mut()
            }
        }

        impl<T: core::cmp::Ord, const N: usize> core::cmp::Ord for $name {
            #[inline]
            fn cmp(&self, other: &Self) -> core::cmp::Ordering {
                core::cmp::Ord::cmp(self.as_slice(), other.as_slice())
            }
        }

        impl<T: core::cmp::PartialOrd, const N: usize> core::cmp::PartialOrd for $name {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                core::cmp::PartialOrd::partial_cmp(self.as_slice(), other.as_slice())
            }
        }

        impl<T: Eq, const N: usize> Eq for $name {}

        impl<T, U, const N: usize> core::cmp::PartialEq<[U]> for $name
        where
            T: core::cmp::PartialEq<U>,
        {
            #[inline]
            fn eq(&self, other: &[U]) -> bool {
                core::cmp::PartialEq::eq(self.as_slice(), other)
            }
        }

        impl<T, U, const N: usize> core::cmp::PartialEq<&[U]> for $name
        where
            T: core::cmp::PartialEq<U>,
        {
            #[inline]
            fn eq(&self, other: &&[U]) -> bool {
                core::cmp::PartialEq::eq(self.as_slice(), *other)
            }
        }

        impl<T, U, const N: usize> core::cmp::PartialEq<&mut [U]> for $name
        where
            T: core::cmp::PartialEq<U>,
        {
            #[inline]
            fn eq(&self, other: &&mut [U]) -> bool {
                core::cmp::PartialEq::eq(self.as_slice(), *other)
            }
        }

        impl<T, U, const N: usize, const P: usize> core::cmp::PartialEq<[U; P]> for $name
        where
            T: core::cmp::PartialEq<U>,
        {
            #[inline]
            fn eq(&self, other: &[U; P]) -> bool {
                core::cmp::PartialEq::eq(self.as_slice(), other.as_slice())
            }
        }

        impl<T, U, const N: usize, const P: usize> core::cmp::PartialEq<&[U; P]> for $name
        where
            T: core::cmp::PartialEq<U>,
        {
            #[inline]
            fn eq(&self, other: &&[U; P]) -> bool {
                core::cmp::PartialEq::eq(self.as_slice(), other.as_slice())
            }
        }
    };
}

pub(super) use impl_common_traits;
