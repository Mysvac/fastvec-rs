extern crate std;

use core::ptr;
use std::io::{IoSlice, Write};

use crate::{AutoVec, FastVec, StackVec};

/// Write is implemented for `StackVec<u8, N>` by appending to the vector.
///
/// If the vector is full, [`Write::write`] will return `Ok(0)`.
#[cfg(feature = "std")]
impl<const N: usize> Write for StackVec<u8, N> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let len = self.len;
        let num = core::cmp::min(N - len, buf.len());

        unsafe {
            ptr::copy_nonoverlapping(buf.as_ptr(), self.as_mut_ptr().add(len), num);
            self.set_len(len + num);
        }

        Ok(num)
    }

    #[inline(always)]
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }

    #[inline]
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> std::io::Result<usize> {
        let mut num = 0;
        for buf in bufs {
            if self.is_full() {
                break;
            }
            num += self.write(buf)?;
        }
        Ok(num)
    }
}

/// Write is implemented for `FastVec<u8, N>` by appending to the vector.
/// The vector will grow as needed.
#[cfg(feature = "std")]
impl<const N: usize> Write for FastVec<u8, N> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let len = self.len();
        let num = buf.len();

        let vec = self.data();

        vec.reserve(num);

        unsafe {
            ptr::copy_nonoverlapping(buf.as_ptr(), vec.as_mut_ptr().add(len), num);
            vec.set_len(len + num);
        }

        Ok(num)
    }

    #[inline(always)]
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }

    #[inline]
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> std::io::Result<usize> {
        let num = bufs.iter().map(|b| b.len()).sum::<usize>();

        let vec = self.data();
        vec.reserve(num);
        for buf in bufs {
            let buf_len = buf.len();
            let vec_len = vec.len();
            unsafe {
                ptr::copy_nonoverlapping(buf.as_ptr(), vec.as_mut_ptr().add(vec_len), buf_len);
                vec.set_len(vec_len + buf_len);
            }
        }

        Ok(num)
    }

    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        Write::write(self, buf)?;
        Ok(())
    }
}

/// Write is implemented for `AutoVec<u8, N>` by appending to the vector.
/// The vector will grow as needed.
#[cfg(feature = "std")]
impl<const N: usize> Write for AutoVec<u8, N> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let len = self.len();
        let num = buf.len();

        self.reserve(num);

        unsafe {
            ptr::copy_nonoverlapping(buf.as_ptr(), self.as_mut_ptr().add(len), num);
            self.set_len(len + num);
        }

        Ok(num)
    }

    #[inline(always)]
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }

    #[inline]
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> std::io::Result<usize> {
        let num = bufs.iter().map(|b| b.len()).sum::<usize>();

        self.reserve(num);
        let base_ptr = self.as_mut_ptr();
        for buf in bufs {
            let buf_len = buf.len();
            let vec_len = self.len();
            unsafe {
                ptr::copy_nonoverlapping(buf.as_ptr(), base_ptr.add(vec_len), buf_len);
                self.set_len(vec_len + buf_len);
            }
        }

        Ok(num)
    }

    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        Write::write(self, buf)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{IoSlice, Write};

    #[test]
    fn stackvec_write_basic_and_partial() {
        let mut v: StackVec<u8, 4> = StackVec::new();
        // First write fits entirely
        let n = v.write(b"ab").unwrap();
        assert_eq!(n, 2);
        assert_eq!(v.len(), 2);
        assert_eq!(v, b"ab");

        // Second write partially fits
        let n = v.write(b"cdef").unwrap();
        assert_eq!(n, 2); // only 'c','d' fit
        assert_eq!(v.len(), 4);
        assert_eq!(v, b"abcd");

        // Full
        assert_eq!(v.write(b"abcd").unwrap(), 0);
        assert_eq!(v, b"abcd");
    }

    #[test]
    fn stackvec_write_vectored_partial_stop_on_full() {
        let mut v: StackVec<u8, 5> = StackVec::new();
        let bufs = [
            IoSlice::new(b"ab"),
            IoSlice::new(b"cde"),
            IoSlice::new(b"f"),
        ];
        let n = v.write_vectored(&bufs).unwrap();
        assert_eq!(n, 5);
        assert_eq!(v, b"abcde");
    }

    #[test]
    fn fastvec_write_and_vectored() {
        let mut v: FastVec<u8, 4> = FastVec::new();

        let n = v.write(b"hello").unwrap();
        assert_eq!(n, 5);
        assert_eq!(v.len(), 5);
        assert_eq!(v, b"hello");

        let bufs = [IoSlice::new(b" "), IoSlice::new(b"world")];
        let n = v.write_vectored(&bufs).unwrap();
        assert_eq!(n, 6);
        assert_eq!(v, b"hello world");
    }

    #[test]
    fn fastvec_write_all_grows() {
        let mut v: FastVec<u8, 3> = FastVec::new();
        let data = [b'x'; 257];
        v.write_all(&data).unwrap();
        assert_eq!(v.len(), 257);
        assert!(v.as_slice().iter().all(|&c| c == b'x'));
    }

    #[test]
    fn autovec_write_and_vectored() {
        let mut v: AutoVec<u8, 4> = AutoVec::new();

        let n = v.write(b"hello").unwrap();
        assert_eq!(n, 5);
        assert_eq!(v.len(), 5);
        assert_eq!(v, b"hello");

        let bufs = [IoSlice::new(b" "), IoSlice::new(b"world")];
        let n = v.write_vectored(&bufs).unwrap();
        assert_eq!(n, 6);
        assert_eq!(v, b"hello world");
    }

    #[test]
    fn autovec_write_all_grows() {
        let mut v: AutoVec<u8, 3> = AutoVec::new();
        let data = [b'y'; 257];
        v.write_all(&data).unwrap();
        assert_eq!(v.len(), 257);
        assert!(v.as_slice().iter().all(|&c| c == b'y'));
    }
}
