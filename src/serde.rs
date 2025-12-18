use crate::{AutoVec, FastVec, StackVec};
use alloc::format;
use core::marker::PhantomData;
use serde_core::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self, SeqAccess, Visitor},
    ser::SerializeSeq,
};

#[cfg(feature = "serde")]
impl<T: Serialize, const N: usize> Serialize for StackVec<T, N> {
    /// Serialize a [`StackVec`] as a sequence.
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

#[cfg(feature = "serde")]
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

#[cfg(feature = "serde")]
impl<T: Serialize, const N: usize> Serialize for AutoVec<T, N> {
    /// Serialize an [`AutoVec`] as a sequence.
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

#[cfg(feature = "serde")]
impl<'de, T: Deserialize<'de>, const N: usize> Deserialize<'de> for StackVec<T, N> {
    /// Deserialize a [`StackVec`] from a sequence.
    ///
    /// # Panics
    /// Panics if the sequence length exceeds the stack capacity `N`.
    #[inline]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct StackVecVisitor<T, const N: usize> {
            _marker: PhantomData<T>,
        }

        impl<'de, T: Deserialize<'de>, const N: usize> Visitor<'de> for StackVecVisitor<T, N> {
            type Value = StackVec<T, N>;

            fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                formatter.write_str("a sequence")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut vec = StackVec::new();

                while let Some(element) = seq.next_element()? {
                    if vec.len() < N {
                        vec.push(element);
                    } else {
                        return Err(de::Error::custom(format!(
                            "StackVec capacity {} exceeded while deserializing sequence",
                            N
                        )));
                    }
                }

                Ok(vec)
            }
        }

        deserializer.deserialize_seq(StackVecVisitor {
            _marker: PhantomData,
        })
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Deserialize<'de>, const N: usize> Deserialize<'de> for FastVec<T, N> {
    /// Deserialize an [`FastVec`] from a sequence.
    ///
    /// If the sequence length exceeds the stack capacity `N`, the data will be stored on the heap.
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

                let vec_mut = vec.get();

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

#[cfg(feature = "serde")]
impl<'de, T: Deserialize<'de>, const N: usize> Deserialize<'de> for AutoVec<T, N> {
    /// Deserialize an [`AutoVec`] from a sequence.
    ///
    /// If the sequence length exceeds the stack capacity `N`, the data will be stored on the heap.
    #[inline]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct AutoVecVisitor<T, const N: usize> {
            _marker: PhantomData<T>,
        }

        impl<'de, T: Deserialize<'de>, const N: usize> Visitor<'de> for AutoVecVisitor<T, N> {
            type Value = AutoVec<T, N>;

            fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                formatter.write_str("a sequence")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut vec = match seq.size_hint() {
                    Some(hint) => AutoVec::with_capacity(hint),
                    None => AutoVec::new(),
                };

                while let Some(element) = seq.next_element::<T>()? {
                    vec.push(element);
                }

                Ok(vec)
            }
        }

        deserializer.deserialize_seq(AutoVecVisitor {
            _marker: PhantomData,
        })
    }
}
