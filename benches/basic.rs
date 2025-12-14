use criterion::{Criterion, criterion_group, criterion_main};
use fastvec::{AutoVec, StackVec};
use smallvec::SmallVec;
use std::hint;

const VEC_SIZE: usize = 16;

trait VecLike<T> {
    fn new_with_small_capacity() -> Self;
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
            self.pop();
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
    };
    (none) => {
        impl_vec_like!();
        #[inline]
        fn new_with_small_capacity() -> Self {
            Self::new()
        }
    };
}

impl<T> VecLike<T> for Vec<T> {
    impl_vec_like! {with}
}
impl<T> VecLike<T> for SmallVec<T, VEC_SIZE> {
    impl_vec_like! {none}
}
impl<T> VecLike<T> for StackVec<T, VEC_SIZE> {
    impl_vec_like! {none}
}
impl<T> VecLike<T> for AutoVec<T, VEC_SIZE> {
    impl_vec_like! {none}
}

macro_rules! gen_benches {
    (Common: $handle:ident, $name:literal => $func:ident) => {{
        let mut group_new = $handle.benchmark_group($name);
        group_new.bench_function("Vec", |b| std::hint::black_box($func::<u64, Vec<u64>>(b)));
        group_new.bench_function("AutoVec", |b| {
            std::hint::black_box($func::<u64, AutoVec<u64, VEC_SIZE>>(b))
        });
        group_new.bench_function("SmallVec", |b| {
            std::hint::black_box($func::<u64, SmallVec<u64, VEC_SIZE>>(b))
        });
    }};
    (All: $handle:ident, $name:literal => $func:ident) => {{
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
}

fn bench_vec(c: &mut Criterion) {
    gen_benches!(All: c, "new_with_small_capacity" => new);
    gen_benches!(All: c, "push_not_overflow" => push_mini);
    gen_benches!(All: c, "pop_mini" => pop_mini);
    gen_benches!(All: c, "insert_not_overflow" => insert_mini);
    gen_benches!(All: c, "remove_mini" => remove_mini);
}

fn new<T, V: VecLike<T>>(b: &mut criterion::Bencher) {
    b.iter(|| hint::black_box(V::new_with_small_capacity()));
}

fn push_mini<T: Default, V: VecLike<T>>(b: &mut criterion::Bencher) {
    let mut v = V::new_with_small_capacity();
    b.iter(|| {
        hint::black_box({
            for _ in 0..16 {
                hint::black_box(v.push(T::default()));
            }
            v.set_len(0);
        })
    });
    v.set_len(0);
}

fn pop_mini<T, V: VecLike<T>>(b: &mut criterion::Bencher) {
    let mut v = V::new_with_small_capacity();
    b.iter(|| {
        hint::black_box({
            v.set_len(16);
            for _ in 0..16 {
                hint::black_box(v.pop());
            }
        })
    });
    v.set_len(0);
}

fn insert_mini<T: Default, V: VecLike<T>>(b: &mut criterion::Bencher) {
    let mut v = V::new_with_small_capacity();
    b.iter(|| {
        hint::black_box({
            v.set_len(12);
            hint::black_box(v.insert(0, T::default()));
            hint::black_box(v.insert(7, T::default()));
            hint::black_box(v.insert(14, T::default()));
        })
    });
    v.set_len(0);
}

fn remove_mini<T: Default, V: VecLike<T>>(b: &mut criterion::Bencher) {
    let mut v = V::new_with_small_capacity();
    b.iter(|| {
        hint::black_box({
            v.set_len(14);
            hint::black_box(v.remove(13));
            hint::black_box(v.remove(7));
            hint::black_box(v.remove(0));
        })
    });
    v.set_len(0);
}

criterion_group!(benches, bench_vec);
criterion_main!(benches);
