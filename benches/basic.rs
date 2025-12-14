use criterion::{Criterion, criterion_group, criterion_main};
use fastvec::{AutoVec, StackVec};
use smallvec::SmallVec;
use std::hint;

const VEC_SIZE: usize = 16;
const LARGE_SIZE: usize = 40000;

trait VecLike<T> {
    fn new_with_small_capacity() -> Self;
    fn new_with_large_capacity() -> Self;
    fn push(&mut self, value: T);
    fn pop(&mut self);
    fn insert(&mut self, index: usize, value: T);
    fn remove(&mut self, index: usize);
    fn set_len(&mut self, len: usize);
}

macro_rules! impl_vec_like {
    () => {
        #[inline]
        fn push(&mut self, value: T) {
            self.push(value);
        }
        #[inline]
        fn pop(&mut self) {
            let _ = self.pop();
        }
        #[inline]
        fn insert(&mut self, index: usize, value: T) {
            self.insert(index, value);
        }
        #[inline]
        fn remove(&mut self, index: usize) {
            self.remove(index);
        }
        #[inline(always)]
        fn set_len(&mut self, len: usize) {
            unsafe {
                Self::set_len(self, len);
            }
        }
    };
    (with) => {
        impl_vec_like!();
        #[inline]
        fn new_with_small_capacity() -> Self {
            Self::with_capacity(VEC_SIZE)
        }
        #[inline]
        fn new_with_large_capacity() -> Self {
            Self::with_capacity(LARGE_SIZE)
        }
    };
    (none) => {
        impl_vec_like!();
        #[inline]
        fn new_with_small_capacity() -> Self {
            Self::new()
        }
        #[inline]
        fn new_with_large_capacity() -> Self {
            Self::new()
        }
    };
}

impl<T> VecLike<T> for Vec<T> {
    impl_vec_like! {with}
}
impl<T> VecLike<T> for SmallVec<T, VEC_SIZE> {
    impl_vec_like! {with}
}
impl<T> VecLike<T> for AutoVec<T, VEC_SIZE> {
    impl_vec_like! {with}
}
impl<T> VecLike<T> for StackVec<T, VEC_SIZE> {
    impl_vec_like! {none}
}
impl<T> VecLike<T> for StackVec<T, LARGE_SIZE> {
    impl_vec_like! {none}
}

macro_rules! gen_benches {
    (Large: $handle:ident, $name:literal => $func:ident) => {{
        let mut group_new = $handle.benchmark_group($name);
        group_new.bench_function("Vec", |b| std::hint::black_box($func::<u64, Vec<u64>>(b)));
        group_new.bench_function("AutoVec", |b| {
            std::hint::black_box($func::<u64, AutoVec<u64, VEC_SIZE>>(b))
        });
        group_new.bench_function("SmallVec", |b| {
            std::hint::black_box($func::<u64, SmallVec<u64, VEC_SIZE>>(b))
        });
        group_new.bench_function("StackVec", |b| {
            std::hint::black_box($func::<u64, StackVec<u64, LARGE_SIZE>>(b))
        });
    }};
    (Small: $handle:ident, $name:literal => $func:ident) => {{
        let mut group_new = $handle.benchmark_group($name);
        group_new.bench_function("Vec", |b| std::hint::black_box($func::<u64, Vec<u64>>(b)));
        group_new.bench_function("AutoVec", |b| {
            std::hint::black_box($func::<u64, AutoVec<u64, VEC_SIZE>>(b))
        });
        group_new.bench_function("SmallVec", |b| {
            std::hint::black_box($func::<u64, SmallVec<u64, VEC_SIZE>>(b))
        });
        group_new.bench_function("StackVec", |b| {
            std::hint::black_box($func::<u64, StackVec<u64, VEC_SIZE>>(b))
        });
    }};
    (HeapOnly: $handle:ident, $name:literal => $func:ident) => {{
        let mut group_new = $handle.benchmark_group($name);
        group_new.bench_function("Vec", |b| std::hint::black_box($func::<u64, Vec<u64>>(b)));
        group_new.bench_function("AutoVec", |b| {
            std::hint::black_box($func::<u64, AutoVec<u64, VEC_SIZE>>(b))
        });
        group_new.bench_function("SmallVec", |b| {
            std::hint::black_box($func::<u64, SmallVec<u64, VEC_SIZE>>(b))
        });
    }};
}

fn bench_vec(c: &mut Criterion) {
    gen_benches!(Small: c, "new_with_small_capacity" => new);
    gen_benches!(Small: c, "push_mini" => push_mini);
    gen_benches!(Small: c, "pop_mini" => pop_mini);
    gen_benches!(Small: c, "insert_mini" => insert_mini);
    gen_benches!(Small: c, "remove_mini" => remove_mini);
    gen_benches!(Large: c, "push_large" => push_large);
    gen_benches!(Large: c, "pop_large" => pop_large);
    gen_benches!(Large: c, "insert_large" => insert_large);
    gen_benches!(Large: c, "remove_large" => remove_large);
    gen_benches!(HeapOnly: c, "new_with_large_capacity" => new_large);
    gen_benches!(HeapOnly: c, "push_overflow_small" => push_overflow_small);
    gen_benches!(HeapOnly: c, "push_overflow_large" => push_overflow_large);
}

fn new<T, V: VecLike<T>>(b: &mut criterion::Bencher) {
    b.iter(|| hint::black_box(V::new_with_small_capacity()));
}

fn new_large<T, V: VecLike<T>>(b: &mut criterion::Bencher) {
    b.iter(|| hint::black_box(V::new_with_large_capacity()));
}

fn push_mini<T: Default, V: VecLike<T>>(b: &mut criterion::Bencher) {
    let mut v = V::new_with_small_capacity();
    b.iter(|| {
        hint::black_box({
            hint::black_box(v.set_len(0));
            hint::black_box(v.push(T::default()));
            hint::black_box(v.push(T::default()));
            hint::black_box(v.push(T::default()));
            hint::black_box(v.push(T::default()));
            hint::black_box(v.push(T::default()));
            hint::black_box(v.push(T::default()));
            hint::black_box(v.push(T::default()));
            hint::black_box(v.push(T::default()));
            hint::black_box(v.push(T::default()));
            hint::black_box(v.push(T::default()));
            hint::black_box(v.push(T::default()));
            hint::black_box(v.push(T::default()));
            hint::black_box(v.push(T::default()));
            hint::black_box(v.push(T::default()));
        })
    });
    v.set_len(0);
}

fn pop_mini<T, V: VecLike<T>>(b: &mut criterion::Bencher) {
    let mut v = V::new_with_small_capacity();
    b.iter(|| {
        hint::black_box({
            hint::black_box(v.set_len(16));
            hint::black_box(v.pop());
            hint::black_box(v.pop());
            hint::black_box(v.pop());
            hint::black_box(v.pop());
            hint::black_box(v.pop());
            hint::black_box(v.pop());
            hint::black_box(v.pop());
            hint::black_box(v.pop());
            hint::black_box(v.pop());
            hint::black_box(v.pop());
            hint::black_box(v.pop());
            hint::black_box(v.pop());
            hint::black_box(v.pop());
        })
    });
    v.set_len(0);
}

fn insert_mini<T: Default, V: VecLike<T>>(b: &mut criterion::Bencher) {
    let mut v = V::new_with_small_capacity();
    b.iter(|| {
        hint::black_box({
            hint::black_box(v.set_len(7));
            hint::black_box(v.insert(0, T::default()));
            hint::black_box(v.insert(3, T::default()));
            hint::black_box(v.insert(9, T::default()));
            hint::black_box(v.insert(10, T::default()));
            hint::black_box(v.insert(10, T::default()));
            hint::black_box(v.insert(9, T::default()));
            hint::black_box(v.insert(10, T::default()));
            hint::black_box(v.insert(9, T::default()));
            hint::black_box(v.insert(10, T::default()));
        })
    });
    v.set_len(0);
}

fn remove_mini<T: Default, V: VecLike<T>>(b: &mut criterion::Bencher) {
    let mut v = V::new_with_small_capacity();
    b.iter(|| {
        hint::black_box({
            hint::black_box(v.set_len(16));
            hint::black_box(v.remove(15));
            hint::black_box(v.remove(8));
            hint::black_box(v.remove(0));
            hint::black_box(v.remove(9));
            hint::black_box(v.remove(9));
            hint::black_box(v.remove(6));
            hint::black_box(v.remove(6));
            hint::black_box(v.remove(6));
            hint::black_box(v.remove(2));
            hint::black_box(v.remove(2));
            hint::black_box(v.remove(2));
        })
    });
    v.set_len(0);
}

fn push_large<T: Default, V: VecLike<T>>(b: &mut criterion::Bencher) {
    let mut v = V::new_with_large_capacity();
    b.iter(|| {
        hint::black_box({
            hint::black_box(v.set_len(0));
            for _ in 0..LARGE_SIZE {
                hint::black_box(v.push(T::default()));
            }
        })
    });
    v.set_len(0);
}

fn pop_large<T, V: VecLike<T>>(b: &mut criterion::Bencher) {
    let mut v = V::new_with_large_capacity();
    b.iter(|| {
        hint::black_box({
            hint::black_box(v.set_len(LARGE_SIZE));
            for _ in 0..LARGE_SIZE {
                hint::black_box(v.pop());
            }
        })
    });
    v.set_len(0);
}

fn insert_large<T: Default, V: VecLike<T>>(b: &mut criterion::Bencher) {
    let mut v = V::new_with_large_capacity();
    b.iter(|| {
        hint::black_box({
            hint::black_box(v.set_len(LARGE_SIZE >> 1));
            hint::black_box(v.insert(LARGE_SIZE >> 1, T::default()));
            hint::black_box(v.insert(LARGE_SIZE >> 3, T::default()));
            hint::black_box(v.insert(LARGE_SIZE >> 5, T::default()));
            hint::black_box(v.insert(LARGE_SIZE >> 7, T::default()));
            hint::black_box(v.insert(LARGE_SIZE >> 9, T::default()));
            hint::black_box(v.insert(LARGE_SIZE >> 11, T::default()));
            hint::black_box(v.insert(LARGE_SIZE >> 13, T::default()));
            hint::black_box(v.insert(0, T::default()));
        })
    });
    v.set_len(0);
}

fn remove_large<T: Default, V: VecLike<T>>(b: &mut criterion::Bencher) {
    let mut v = V::new_with_large_capacity();
    b.iter(|| {
        hint::black_box({
            hint::black_box(v.set_len((LARGE_SIZE >> 1) + 20));
            hint::black_box(v.remove(LARGE_SIZE >> 1));
            hint::black_box(v.remove(LARGE_SIZE >> 3));
            hint::black_box(v.remove(LARGE_SIZE >> 5));
            hint::black_box(v.remove(LARGE_SIZE >> 7));
            hint::black_box(v.remove(LARGE_SIZE >> 9));
            hint::black_box(v.remove(LARGE_SIZE >> 11));
            hint::black_box(v.remove(LARGE_SIZE >> 13));
            hint::black_box(v.remove(0));
        })
    });
    v.set_len(0);
}

fn push_overflow_small<T: Default, V: VecLike<T>>(b: &mut criterion::Bencher) {
    b.iter(|| {
        hint::black_box({
            let mut v = V::new_with_small_capacity();
            for _ in 0..VEC_SIZE * 7 {
                hint::black_box(v.push(T::default()));
            }
        })
    });
}

fn push_overflow_large<T: Default, V: VecLike<T>>(b: &mut criterion::Bencher) {
    b.iter(|| {
        hint::black_box({
            let mut v = V::new_with_small_capacity();
            for _ in 0..LARGE_SIZE {
                hint::black_box(v.push(T::default()));
            }
        })
    });
}

criterion_group!(benches, bench_vec);
criterion_main!(benches);
