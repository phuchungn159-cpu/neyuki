//! A small, fast, non-cryptographic hasher for the VM's string-keyed maps.
//!
//! Table fields and globals are looked up on nearly every instruction that
//! touches a name, and the standard SipHash is built to resist collision
//! attacks rather than to be quick on short keys. This is the FxHash scheme
//! used inside rustc: a multiply-and-rotate per word. Keys never come from an
//! untrusted network source in a way that needs DoS resistance.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

#[derive(Default, Clone, Copy)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline(always)]
    fn add_to_hash(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let len = bytes.len();
        if len == 0 {
            return;
        }
        if len < 8 {
            let word = match len {
                1 => bytes[0] as u64,
                2 => u16::from_le_bytes([bytes[0], bytes[1]]) as u64,
                3 => (bytes[0] as u64) | ((bytes[1] as u64) << 8) | ((bytes[2] as u64) << 16),
                4 => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64,
                5 => {
                    let lo = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64;
                    self.add_to_hash(lo);
                    bytes[4] as u64
                }
                6 => {
                    let lo = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64;
                    self.add_to_hash(lo);
                    u16::from_le_bytes([bytes[4], bytes[5]]) as u64
                }
                _ => {
                    let lo = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64;
                    self.add_to_hash(lo);
                    (bytes[4] as u64) | ((bytes[5] as u64) << 8) | ((bytes[6] as u64) << 16)
                }
            };
            self.add_to_hash(word);
            return;
        }

        let (chunks, rest) = bytes.as_chunks::<8>();
        for chunk in chunks {
            self.add_to_hash(u64::from_le_bytes(*chunk));
        }
        if rest.len() >= 4 {
            self.add_to_hash(u32::from_le_bytes([rest[0], rest[1], rest[2], rest[3]]) as u64);
            for &b in &rest[4..] {
                self.add_to_hash(b as u64);
            }
        } else {
            for &b in rest {
                self.add_to_hash(b as u64);
            }
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add_to_hash(i as u64);
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add_to_hash(i as u64);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add_to_hash(i as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add_to_hash(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add_to_hash(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        // The multiply leaves the entropy in the high bits, but the table
        // picks buckets from the low ones: aligned pointers and small
        // integers would otherwise all share a handful of buckets.
        self.hash.rotate_left(26)
    }
}

pub type FxBuildHasher = BuildHasherDefault<FxHasher>;
pub type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;
pub type FxHashSet<K> = std::collections::HashSet<K, FxBuildHasher>;

/// `FxHashMap::new()` is not available because the hasher is not
/// `RandomState`; this is the equivalent constructor.
pub fn new_map<K, V>() -> FxHashMap<K, V> {
    HashMap::with_hasher(FxBuildHasher::default())
}

/// Constructs a `FxHashMap` with preallocated capacity.
pub fn with_capacity<K, V>(capacity: usize) -> FxHashMap<K, V> {
    HashMap::with_capacity_and_hasher(capacity, FxBuildHasher::default())
}

pub fn new_set<K>() -> FxHashSet<K> {
    std::collections::HashSet::with_hasher(FxBuildHasher::default())
}

pub fn set_with_capacity<K>(capacity: usize) -> FxHashSet<K> {
    std::collections::HashSet::with_capacity_and_hasher(capacity, FxBuildHasher::default())
}

#[inline]
pub fn fx_hash_bytes(bytes: &[u8]) -> u64 {
    let mut h = FxHasher::default();
    h.write(bytes);
    h.finish()
}
