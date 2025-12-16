//! See `README.md`

use core::hint;
use criterion::{Bencher, Criterion, criterion_group, criterion_main};
use fastvec::{FastVec, StackVec, fast_vec::FastVecData};
use smallvec::SmallVec;
use std::sync::OnceLock;

use rand::Rng;

const SMALL_SIZE: usize = 16;
const SMALL_SIZE_1: usize = 17;
const LARGE_SIZE: usize = 40000;

/// A function used to generate a random amount of data.
///
/// We use random data to simulate real-world scenarios and
/// avoid excessive optimization by the compiler when it knows the context.
///
/// Note: If the data is not random and the function is inline expanded,
/// a large amount of code will be deleted due to compile time optimization,
/// resulting in completely different test results from actual results.
#[inline(never)]
fn gen_one(start: usize, end: usize) -> usize {
    let mut rng = rand::rng();
    rng.random_range(start..end)
}

/// The amount of data used in small data testing,
/// is randomly generated to avoid the compiler optimizing based on accurate data volume.
///
/// This is reasonable, as scenarios using vectors usually do not know the amount of data.
static SMALL_BOUND: OnceLock<usize> = OnceLock::new();

/// The amount of data used in large data testing,
/// is randomly generated to avoid the compiler optimizing based on accurate data volume.
///
/// This is reasonable, as scenarios using vectors usually do not know the amount of data.
static LARGE_BOUND: OnceLock<usize> = OnceLock::new();

/// Generate an array of random content of a specified length.
///
/// We use random data to simulate real-world scenarios and
/// avoid excessive optimization by the compiler when it knows the context.
///
/// Note: If the data is not random and the function is inline expanded,
/// a large amount of code will be deleted due to compile time optimization,
/// resulting in completely different test results from actual results.
#[inline(never)]
fn gen_rand(len: usize, start: u64, end: u64) -> Box<[u64]> {
    let mut rng = rand::rng();
    let mut vec: Vec<u64> = Vec::with_capacity(len);
    for _ in 0..len {
        vec.push(rng.random_range(start..end));
    }
    vec.into_boxed_slice()
}

/// An initialization tool for vector like type.
trait VecLike {
    fn new_empty() -> Self;
    fn new_small() -> Self;
    fn new_large() -> Self;
    fn accessor(&mut self) -> impl Accessor<'_>;
}

/// A data manipulation tool for vectors.
///
/// Due to the particularity of [`FastVec`] usage, we have to
/// separate initialization and data manipulation.
///
/// For other types, this is equivalent to a mutable reference of itself.
trait Accessor<'a> {
    fn push(&mut self, value: u64);
    fn pop(&mut self) -> Option<u64>;
    fn insert(&mut self, index: usize, value: u64);
    fn remove(&mut self, index: usize) -> u64;
    fn get_mut(&mut self, index: usize) -> &mut u64;
    /// Used for quickly setting vector contents during testing.
    ///
    /// We use u64 testing and do not need to call [`Drop`].
    fn set_len(&mut self, len: usize);
}

macro_rules! impl_accessor {
    ($name:ty) => {
        impl<'a> Accessor<'a> for &'a mut $name {
            #[inline(always)]
            fn push(&mut self, value: u64) {
                (*self).push(value)
            }
            #[inline(always)]
            fn pop(&mut self) -> Option<u64> {
                (*self).pop()
            }
            #[inline(always)]
            fn insert(&mut self, index: usize, value: u64) {
                (*self).insert(index, value);
            }
            #[inline(always)]
            fn remove(&mut self, index: usize) -> u64 {
                (*self).remove(index)
            }
            #[inline(always)]
            fn get_mut(&mut self, index: usize) -> &mut u64 {
                &mut (*self)[index]
            }
            #[inline(always)]
            fn set_len(&mut self, len: usize) {
                unsafe {
                    (*self).set_len(len);
                }
            }
        }
    };
}

impl VecLike for Vec<u64> {
    #[inline(always)]
    fn new_empty() -> Self {
        Self::new()
    }
    #[inline(always)]
    fn new_small() -> Self {
        Self::with_capacity(SMALL_SIZE)
    }
    #[inline(always)]
    fn new_large() -> Self {
        Self::with_capacity(LARGE_SIZE)
    }
    #[inline(always)]
    fn accessor(&mut self) -> impl Accessor<'_> {
        self
    }
}

impl_accessor!(Vec<u64>);

impl VecLike for StackVec<u64, SMALL_SIZE> {
    #[inline(always)]
    fn new_empty() -> Self {
        Self::new()
    }
    #[inline(always)]
    fn new_small() -> Self {
        Self::new()
    }
    #[inline(always)]
    fn new_large() -> Self {
        unreachable!()
    }
    #[inline(always)]
    fn accessor(&mut self) -> impl Accessor<'_> {
        self
    }
}

impl_accessor!(StackVec<u64, SMALL_SIZE>);

impl VecLike for SmallVec<u64, SMALL_SIZE> {
    #[inline(always)]
    fn new_empty() -> Self {
        Self::new()
    }
    #[inline(always)]
    fn new_small() -> Self {
        Self::new()
    }
    #[inline(always)]
    fn new_large() -> Self {
        Self::with_capacity(LARGE_SIZE)
    }
    #[inline(always)]
    fn accessor(&mut self) -> impl Accessor<'_> {
        self
    }
}

impl_accessor!(SmallVec<u64, SMALL_SIZE>);

/// This is a normal usage test for [`FastVec`].
///
/// First, obtain the handle through [`get_mut`](FastVec::get_mut),
/// and then perform efficient operations through the handle.
impl VecLike for FastVec<u64, SMALL_SIZE> {
    #[inline(always)]
    fn new_empty() -> Self {
        Self::new()
    }
    #[inline(always)]
    fn new_small() -> Self {
        Self::new()
    }
    #[inline(always)]
    fn new_large() -> Self {
        Self::with_capacity(LARGE_SIZE)
    }
    #[inline(always)]
    fn accessor(&mut self) -> impl Accessor<'_> {
        self.get_mut()
    }
}

impl_accessor!(FastVecData<u64, SMALL_SIZE>);

/// This is a incorrect usage test for [`FastVec`].
///
/// Never obtain handles, call [`get_mut`](FastVec::get_mut) or
/// [`get_ref`](FastVec::get_ref) before all operations.
///
/// This will result in an additional branch judgment overhead for all operations,
/// which may also include a pointer assignment.
impl VecLike for FastVec<u64, SMALL_SIZE_1> {
    #[inline(always)]
    fn new_empty() -> Self {
        Self::new()
    }
    #[inline(always)]
    fn new_small() -> Self {
        Self::new()
    }
    #[inline(always)]
    fn new_large() -> Self {
        Self::with_capacity(LARGE_SIZE)
    }
    #[inline(always)]
    fn accessor(&mut self) -> impl Accessor<'_> {
        self
    }
}

impl<'a> Accessor<'a> for &'a mut FastVec<u64, SMALL_SIZE_1> {
    #[inline(always)]
    fn push(&mut self, value: u64) {
        (*self).get_mut().push(value);
    }
    #[inline(always)]
    fn pop(&mut self) -> Option<u64> {
        (*self).get_mut().pop()
    }
    #[inline(always)]
    fn insert(&mut self, index: usize, value: u64) {
        (*self).get_mut().insert(index, value);
    }
    #[inline(always)]
    fn remove(&mut self, index: usize) -> u64 {
        (*self).get_mut().remove(index)
    }
    #[inline(always)]
    fn get_mut(&mut self, index: usize) -> &mut u64 {
        &mut (*self).get_mut()[index]
    }
    #[inline(always)]
    fn set_len(&mut self, len: usize) {
        unsafe {
            (*self).get_mut().set_len(len);
        }
    }
}

macro_rules! gen_bench_group {
    (ExcludeStackVec, $c:ident => $fn_name:ident) => {{
        let mut group_new = $c.benchmark_group(stringify!($fn_name));
        group_new.bench_function("Vec", |b| $fn_name::<Vec<u64>>(b));
        group_new.bench_function("FastVec", |b| $fn_name::<FastVec<u64, SMALL_SIZE>>(b));
        group_new.bench_function("SmallVec", |b| $fn_name::<SmallVec<u64, SMALL_SIZE>>(b));
        group_new.bench_function("FastVec_Direct", |b| {
            $fn_name::<FastVec<u64, SMALL_SIZE_1>>(b)
        });
    }};
    (IncludeStackVec, $c:ident => $fn_name:ident) => {{
        let mut group_new = $c.benchmark_group(stringify!($fn_name));
        group_new.bench_function("Vec", |b| $fn_name::<Vec<u64>>(b));
        group_new.bench_function("FastVec", |b| $fn_name::<FastVec<u64, SMALL_SIZE>>(b));
        group_new.bench_function("StackVec", |b| $fn_name::<StackVec<u64, SMALL_SIZE>>(b));
        group_new.bench_function("SmallVec", |b| $fn_name::<SmallVec<u64, SMALL_SIZE>>(b));
        group_new.bench_function("FastVec_Direct", |b| {
            $fn_name::<FastVec<u64, SMALL_SIZE_1>>(b)
        });
    }};
}

fn bench_vec(c: &mut Criterion) {
    SMALL_BOUND.get_or_init(|| gen_one(14, 16));
    LARGE_BOUND.get_or_init(|| gen_one(36000, 36003));
    gen_bench_group!(IncludeStackVec, c => new_empty);
    gen_bench_group!(IncludeStackVec, c => new_small);
    gen_bench_group!(ExcludeStackVec, c => new_large);
    gen_bench_group!(IncludeStackVec, c => push_small);
    gen_bench_group!(IncludeStackVec, c => push_small_from_empty);
    gen_bench_group!(ExcludeStackVec, c => push_large);
    gen_bench_group!(ExcludeStackVec, c => push_large_from_empty);
    gen_bench_group!(IncludeStackVec, c => pop_small);
    gen_bench_group!(ExcludeStackVec, c => pop_large);
    gen_bench_group!(IncludeStackVec, c => insert_small);
    gen_bench_group!(ExcludeStackVec, c => insert_large);
    gen_bench_group!(IncludeStackVec, c => remove_small);
    gen_bench_group!(ExcludeStackVec, c => remove_large);
    gen_bench_group!(IncludeStackVec, c => index_small);
    gen_bench_group!(ExcludeStackVec, c => index_large);
}

/// Test The creation time of empty vector.
///
/// Usually, this should be equally fast as there is no need to
/// apply for heap memory and can be completed at compilation time.
#[inline(never)]
fn new_empty<T: VecLike>(b: &mut Bencher) {
    b.iter(|| hint::black_box(T::new_empty()));
}

/// Test The creation time of vector with capacity `16`.
///
/// Only vectors need to apply for heap memory.
#[inline(never)]
fn new_small<T: VecLike>(b: &mut Bencher) {
    b.iter(|| hint::black_box(T::new_small()));
}

/// Test The creation time of vector with capacity `40000`.
///
/// Every vectors need to apply for heap memory.
///
/// This test is very inaccurate. The application codes for each container are almost identical,
/// and theoretically the speed should be the same.
///
/// However, there is a lot of randomness in the final time consumption,
#[inline(never)]
fn new_large<T: VecLike>(b: &mut Bencher) {
    b.iter(|| hint::black_box(T::new_large()));
}

/// Pre allocate capacity and only test the efficiency of `push`.
///
/// The data volume is 14-15.
#[inline(never)]
fn push_small<T: VecLike>(b: &mut Bencher) {
    let mut vec = T::new_small();
    let mut op = vec.accessor();
    let data = gen_rand(*SMALL_BOUND.get().unwrap(), 0, 9999);
    let index = gen_one(0, *SMALL_BOUND.get().unwrap());

    b.iter(|| {
        // Randomly collect internal data to avoid
        // compiler optimization of these non output codes.
        let mut counter = 0u64;
        op.set_len(0);
        for item in &data {
            op.push(*item);
        }
        counter += *op.get_mut(index);
        hint::black_box(counter)
    });
    op.set_len(0);
}

/// Not pre allocating heap memory (not actually only `Vec` need to alloc,
/// other containers have sufficient stack memory).
///
/// The data volume is 14-15.
#[inline(never)]
fn push_small_from_empty<T: VecLike>(b: &mut Bencher) {
    let data = gen_rand(*SMALL_BOUND.get().unwrap(), 0, 9999);
    let index = gen_one(0, *SMALL_BOUND.get().unwrap());

    b.iter(|| {
        let mut vec = T::new_empty();
        let mut op = vec.accessor();
        // Randomly collect internal data to avoid
        // compiler optimization of these non output codes.
        let mut counter = 0u64;
        for item in &data {
            op.push(*item);
        }
        counter += *op.get_mut(index);
        op.set_len(0);
        hint::black_box(counter)
    });
}

/// Pre allocate capacity and only test the efficiency of `push`.
///
/// The data volume is 36000-36002.
#[inline(never)]
fn push_large<T: VecLike>(b: &mut Bencher) {
    let mut vec = T::new_large();
    let mut op = vec.accessor();
    let data = gen_rand(*LARGE_BOUND.get().unwrap(), 0, 9999);
    let index = gen_rand(10, 0, *LARGE_BOUND.get().unwrap() as _);

    b.iter(|| {
        // Randomly collect internal data to avoid
        // compiler optimization of these non output codes.
        let mut counter = 0u64;
        op.set_len(0);
        for item in &data {
            op.push(*item);
        }
        for item in &index {
            counter += *op.get_mut(*item as usize);
        }
        hint::black_box(counter)
    });
    op.set_len(0);
}

/// Not pre allocating heap memory, all containers need to be expanded.
/// (The initial stack capacity is 16, except for `Vec`.)
///
/// The data volume is 36000-36002.
#[inline(never)]
fn push_large_from_empty<T: VecLike>(b: &mut Bencher) {
    let data = gen_rand(*LARGE_BOUND.get().unwrap(), 0, 9999);
    let index = gen_rand(10, 0, *LARGE_BOUND.get().unwrap() as _);

    b.iter(|| {
        let mut vec = T::new_empty();
        let mut op = vec.accessor();
        // Randomly collect internal data to avoid
        // compiler optimization of these non output codes.
        let mut counter = 0u64;
        for item in &data {
            op.push(*item);
        }
        for item in &index {
            counter += *op.get_mut(*item as usize);
        }
        op.set_len(0);
        hint::black_box(counter)
    });
}

/// Test `pop` efficient, will not reallocate memory.
///
/// The data volume is 14-15.
#[inline(never)]
fn pop_small<T: VecLike>(b: &mut Bencher) {
    let mut vec = T::new_small();

    let mut op = vec.accessor();
    let num = *SMALL_BOUND.get().unwrap();

    b.iter(|| {
        let mut counter = 0u64;
        op.set_len(num);
        for _ in 1..num {
            unsafe {
                counter += op.pop().unwrap_unchecked();
            }
        }
        hint::black_box(counter)
    });
    op.set_len(0);
}

/// Test `pop` efficient, will not reallocate memory.
///
/// The data volume is 36000-36002.
#[inline(never)]
fn pop_large<T: VecLike>(b: &mut Bencher) {
    let mut vec = T::new_large();

    let mut op = vec.accessor();
    let num = *LARGE_BOUND.get().unwrap();

    b.iter(|| {
        let mut counter = 0u64;
        op.set_len(num);
        for _ in 1..num {
            unsafe {
                counter += op.pop().unwrap_unchecked();
            }
        }
        hint::black_box(counter)
    });
    op.set_len(0);
}

/// Test `insert` efficient, will not reallocate memory.
///
/// The data volume is 36000-36002.
#[inline(never)]
fn insert_small<T: VecLike>(b: &mut Bencher) {
    let mut vec = T::new_small();

    let mut op = vec.accessor();
    let num = *SMALL_BOUND.get().unwrap();
    let index = gen_one(0, 16);

    b.iter(|| {
        let mut counter = 0u64;
        op.set_len(12);
        op.insert({ num + 4 } % 12, 6);
        op.insert({ num + 7 } % 13, 7);
        op.insert({ num + 9 } % 14, 8);
        op.insert({ num + 14 } % 15, 11);
        counter += *op.get_mut(index);
        hint::black_box(counter)
    });
    op.set_len(0);
}

/// Test `insert` efficient, will not reallocate memory.
///
/// The data volume is 36000-36002.
#[inline(never)]
fn insert_large<T: VecLike>(b: &mut Bencher) {
    let mut vec = T::new_large();

    let mut op = vec.accessor();
    let num = *LARGE_BOUND.get().unwrap();
    let index = gen_one(0, 36004);

    b.iter(|| {
        let mut counter = 0u64;
        op.set_len(36000);
        op.insert(num % 12 + 35000, 6);
        op.insert(num % 20 + 20000, 7);
        op.insert(num % 16 + 10000, 8);
        op.insert(num % 13, 11);
        counter += *op.get_mut(index);
        hint::black_box(counter)
    });
    op.set_len(0);
}

/// Test `remove` efficient, will not reallocate memory.
///
/// The data volume is 14-15.
#[inline(never)]
fn remove_small<T: VecLike>(b: &mut Bencher) {
    let mut vec = T::new_small();

    let mut op = vec.accessor();
    let num = *SMALL_BOUND.get().unwrap();
    let index = gen_one(0, 12);

    b.iter(|| {
        let mut counter = 0u64;
        op.set_len(16);
        op.remove({ num + 14 } % 15);
        op.remove({ num + 9 } % 14);
        op.remove({ num + 7 } % 13);
        op.remove({ num + 4 } % 12);
        counter += *op.get_mut(index);
        hint::black_box(counter)
    });
    op.set_len(0);
}

/// Test `insert` efficient, will not reallocate memory.
///
/// The data volume is 36000-36002.
#[inline(never)]
fn remove_large<T: VecLike>(b: &mut Bencher) {
    let mut vec = T::new_large();

    let mut op = vec.accessor();
    let num = *LARGE_BOUND.get().unwrap();
    let index = gen_one(0, 36000);

    b.iter(|| {
        let mut counter = 0u64;
        op.set_len(36050);
        op.remove(num % 12 + 35000);
        op.remove(num % 20 + 20000);
        op.remove(num % 16 + 10000);
        op.remove(num % 13);
        counter += *op.get_mut(index);
        hint::black_box(counter)
    });
    op.set_len(0);
}

/// Test `index` efficient, will not reallocate memory.
///
/// The data volume is 14-15.
#[inline(never)]
fn index_small<T: VecLike>(b: &mut Bencher) {
    let mut vec = T::new_small();
    let mut op = vec.accessor();
    op.set_len(16);

    let index = gen_one(0, 16);
    let range = gen_rand(10, 0, 16);

    b.iter(|| {
        let mut counter = 0u64;
        for item in &range {
            *op.get_mut(*item as usize) += *item;
        }
        counter += *op.get_mut(index);
        hint::black_box(counter)
    });
    op.set_len(0);
}

/// Test `index` efficient, will not reallocate memory.
///
/// The data volume is 36000-36002.
#[inline(never)]
fn index_large<T: VecLike>(b: &mut Bencher) {
    let mut vec = T::new_large();
    let mut op = vec.accessor();
    op.set_len(36000);

    let index = gen_one(0, 36000);
    let range = gen_rand(2000, 0, 36000);

    b.iter(|| {
        let mut counter = 0u64;
        for item in &range {
            *op.get_mut(*item as usize) += *item;
        }
        counter += *op.get_mut(index);
        hint::black_box(counter)
    });
    op.set_len(0);
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(500)
        .warm_up_time(core::time::Duration::from_secs(3))
        .measurement_time(core::time::Duration::from_secs(12))
        .confidence_level(0.96)
        .noise_threshold(0.04);
    targets = bench_vec,
}
criterion_main!(benches);
