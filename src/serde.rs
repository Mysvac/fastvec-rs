use alloc::format;
use core::marker::PhantomData;
use serde_core::de::{self, SeqAccess, Visitor};
use serde_core::ser::SerializeSeq;
use serde_core::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(feature = "arrayvec")]
use crate::ArrayVec;

#[cfg(feature = "fastvec")]
use crate::FastVec;

#[cfg(feature = "smallvec")]
use crate::SmallVec;

#[cfg(all(feature = "arrayvec", feature = "serde"))]
impl<T: Serialize, const N: usize> Serialize for ArrayVec<T, N> {
    /// Serialize a [`ArrayVec`] as a sequence.
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.len()))?;
        for element in self {
            seq.serialize_element(element)?;
        }
        seq.end()
    }
}

#[cfg(all(feature = "fastvec", feature = "serde"))]
impl<T: Serialize, const N: usize> Serialize for FastVec<T, N> {
    /// Serialize a [`FastVec`] as a sequence.
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.len()))?;
        for element in self {
            seq.serialize_element(element)?;
        }
        seq.end()
    }
}

#[cfg(all(feature = "smallvec", feature = "serde"))]
impl<T: Serialize, const N: usize> Serialize for SmallVec<T, N> {
    /// Serialize an [`SmallVec`] as a sequence.
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.len()))?;
        for element in self {
            seq.serialize_element(element)?;
        }
        seq.end()
    }
}

#[cfg(all(feature = "arrayvec", feature = "serde"))]
impl<'de, T: Deserialize<'de>, const N: usize> Deserialize<'de> for ArrayVec<T, N> {
    /// Deserialize a [`ArrayVec`] from a sequence.
    ///
    /// # Panics
    /// Panics if the sequence length exceeds the inline capacity `N`.
    #[inline]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ArrayVecVisitor<T, const N: usize> {
            _marker: PhantomData<T>,
        }

        impl<'de, T: Deserialize<'de>, const N: usize> Visitor<'de> for ArrayVecVisitor<T, N> {
            type Value = ArrayVec<T, N>;

            fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                formatter.write_str("a sequence")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut vec = ArrayVec::new();

                while let Some(element) = seq.next_element()? {
                    if vec.len() < N {
                        vec.push(element);
                    } else {
                        return Err(de::Error::custom(format!(
                            "ArrayVec inline capacity {} exceeded while deserializing sequence",
                            N
                        )));
                    }
                }

                Ok(vec)
            }
        }

        deserializer.deserialize_seq(ArrayVecVisitor {
            _marker: PhantomData,
        })
    }
}

#[cfg(all(feature = "fastvec", feature = "serde"))]
impl<'de, T: Deserialize<'de>, const N: usize> Deserialize<'de> for FastVec<T, N> {
    /// Deserialize an [`FastVec`] from a sequence.
    ///
    /// If the sequence length exceeds the inline capacity `N`, the data will be stored on the heap.
    #[inline]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct FastVecVisitor<T, const N: usize> {
            _marker: PhantomData<T>,
        }

        impl<'de, T: Deserialize<'de>, const N: usize> Visitor<'de> for FastVecVisitor<T, N> {
            type Value = FastVec<T, N>;

            fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                formatter.write_str("a sequence")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut vec = match seq.size_hint() {
                    Some(hint) => FastVec::with_capacity(hint),
                    None => FastVec::new(),
                };

                let vec_mut = vec.data();

                while let Some(element) = seq.next_element::<T>()? {
                    vec_mut.push(element);
                }

                Ok(vec)
            }
        }

        deserializer.deserialize_seq(FastVecVisitor {
            _marker: PhantomData,
        })
    }
}

#[cfg(all(feature = "smallvec", feature = "serde"))]
impl<'de, T: Deserialize<'de>, const N: usize> Deserialize<'de> for SmallVec<T, N> {
    /// Deserialize an [`SmallVec`] from a sequence.
    ///
    /// If the sequence length exceeds the inline capacity `N`, the data will be stored on the heap.
    #[inline]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SmallVecVisitor<T, const N: usize> {
            _marker: PhantomData<T>,
        }

        impl<'de, T: Deserialize<'de>, const N: usize> Visitor<'de> for SmallVecVisitor<T, N> {
            type Value = SmallVec<T, N>;

            fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                formatter.write_str("a sequence")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut vec = match seq.size_hint() {
                    Some(hint) => SmallVec::with_capacity(hint),
                    None => SmallVec::new(),
                };

                while let Some(element) = seq.next_element::<T>()? {
                    vec.push(element);
                }

                Ok(vec)
            }
        }

        deserializer.deserialize_seq(SmallVecVisitor {
            _marker: PhantomData,
        })
    }
}
