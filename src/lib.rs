//! `bin-pool` is a small crate for interning binary slices.
//! A type called [`BinPool`] is provided which represents a collection of binary slices;
//! however, the content is highly optimized for space by allowing slices to overlap in memory.
//! The data is backed by a smaller subset of "large" slices that have been added previously.
//! 
//! Note that interning is not a cheap operation, and in the worst case every backing slice must be examined when adding a new slice.
//! Interning is only recommended when memory is very limited, or for storage optimization in areas where time is not critical (e.g., compiler output).
//! 
//! # Example
//! ```
//! # use bin_pool::*;
//! let mut b = BinPool::new();
//! 
//! b.add(b"hello world".as_slice());  // add first slice - backed by itself
//! b.add(b"hello".as_slice());        // add another slice - backed by first
//! b.add(b"world".as_slice());        // add another slice - backed by first
//! assert_eq!(b.bytes(), 21);         // 21 bytes stored
//! assert_eq!(b.backing_bytes(), 11); // but only 11 bytes needed to represent them
//! 
//! b.add(b"hello world!".as_slice()); // add another slice - becomes the backer for others
//! assert_eq!(b.bytes(), 33);         // now 33 bytes stored
//! assert_eq!(b.backing_bytes(), 12); // but only 12 bytes to represent them
//! ```
//! 
//! # `no_std`
//! 
//! `bin-pool` supports building in `no_std` environments by disabling default features.
//! Note that the `alloc` crate is required.
//! 
//! ```toml
//! [dependencies]
//! bin-pool = { version = "...", default-features = false }
//! ```

#![no_std]
#![forbid(unsafe_code)]

extern crate no_std_compat as std;
use std::prelude::v1::*;
use std::iter::FusedIterator;
use std::borrow::Cow;
use std::slice;

fn find_subregion(sup: &[u8], sub: &[u8]) -> Option<usize> {
    // [T]::windows panics if size is empty, but we know all slices start with empty slice
    if sub.is_empty() { return Some(0) }

    for (i, w) in sup.windows(sub.len()).enumerate() {
        if w == sub { return Some(i) }
    }
    None
}
#[test]
fn test_find_subregion() {
    assert_eq!(find_subregion(&[1, 2, 3], &[1]), Some(0));
    assert_eq!(find_subregion(&[1, 2, 3], &[2]), Some(1));
    assert_eq!(find_subregion(&[1, 2, 3], &[3]), Some(2));
    assert_eq!(find_subregion(&[1, 2, 3, 3, 2, 3, 4], &[2, 3]), Some(1));
    assert_eq!(find_subregion(&[1, 2, 3, 3, 2, 3, 4], &[2, 3, 3]), Some(1));
    assert_eq!(find_subregion(&[1, 2, 3, 3, 2, 3, 4], &[2, 3, 4]), Some(4));
    assert_eq!(find_subregion(&[1, 2, 3, 3, 2, 3, 4], &[2, 3, 4, 5]), None);
    assert_eq!(find_subregion(&[1, 2, 3, 3, 2, 3, 4], &[1, 2, 3, 3, 2, 3, 4]), Some(0));
    assert_eq!(find_subregion(&[1, 2, 3, 3, 2, 3, 4], &[1, 2, 3, 3, 2, 3, 4, 1]), None);
}

/// Information about the location of a [`BinPool`] slice in the backing data.
#[derive(Clone, Copy, Debug)]
pub struct SliceInfo {
    /// Index of the backing slice.
    pub src: usize,
    /// Starting position in the backing slice.
    pub start: usize,
    /// Length of the slice.
    pub length: usize,
}

/// An append-only pool of auto-overlapping binary slices.
/// 
/// In use, you add binary slices one at a time and it will attempt to create maximum overlapping.
/// Currently, it only guarantees that supersets and subsets will overlap.
/// It's theoretically possible to do cross-slice overlapping, but this would be complex and even more expensive.
/// 
/// Slices added to the pool are guaranteed to remain in insertion order.
/// Moreover, the collection is append-only, so once a slice is added it will remain at the same index for the lifetime of the [`BinPool`].
#[derive(Default, Clone, Debug)]
pub struct BinPool {
    data: Vec<Vec<u8>>,     // the backing data
    slices: Vec<SliceInfo>, // effectively slices into top
}
impl BinPool {
    /// Constructs an empty pool.
    pub fn new() -> Self {
        Default::default()
    }

    fn add_internal(&mut self, value: Cow<[u8]>) -> usize {
        // if an equivalent slice already exists, just refer to that
        for (i, other) in self.iter().enumerate() {
            if other == &*value { return i; }
        }

        let ret = self.slices.len(); // eventual return value

        // look for any data entry that value is a subregion of - if we find one we can use that as data source
        for (i, top) in self.data.iter().enumerate() {
            if let Some(start) = find_subregion(top, &*value) {
                self.slices.push(SliceInfo { src: i, start, length: value.len() });
                return ret;
            }
        }

        // if that didn't work, look for any data entry that is a subregion of value (i.e. containment the other way)
        for i in 0..self.data.len() {
            // if we found one, we can replace it with value
            if let Some(start) = find_subregion(&*value, &self.data[i]) {
                // replace it with value and update the starting position of any slices that referenced it
                self.data[i] = value.into_owned();
                for slice in self.slices.iter_mut() {
                    if slice.src == i { slice.start += start; }
                }

                // now we need to look through the data entries again and see if any of them are contained in value (the new, larger data entry)
                // we stopped on first that was a subset of value, so no need to tests 0..=i
                let mut j = i + 1;
                while j < self.data.len() {
                    // if data entry j is contained in value (entry i), we can remove j
                    if let Some(start) = find_subregion(&self.data[i], &self.data[j]) {
                        // get rid of j, redundant with i - use swap remove for efficiency
                        self.data.swap_remove(j);

                        // update all the slices to reflect the change
                        for slice in self.slices.iter_mut() {
                            // if it referenced the deleted entry (j), repoint it to value (i) and apply the start offset
                            if slice.src == j {
                                slice.src = i;
                                slice.start += start;
                            }
                            // if it referenced the moved entry (the one we used for swap remove), repoint it to j
                            else if slice.src == self.data.len() {
                                slice.src = j;
                            }
                        }
                    }
                    else { j += 1; } // only increment j if we didn't remove j
                }

                // and finally, add the slice info
                self.slices.push(SliceInfo { src: i, start: 0, length: self.data[i].len() });
                return ret;
            }
        }

        // if that also didn't work then we just have to add value as a new data entry
        let length = value.len();
        self.data.push(value.into_owned());
        self.slices.push(SliceInfo { src: self.data.len() - 1, start: 0, length });
        ret
    }

    /// Adds the specified slice to the pool.
    /// If an equivalent slice already exists, does nothing and returns the index of the pre-existing slice;
    /// otherwise, adds `value` as a new slice and returns its (new) slice index.
    /// 
    /// If you are working with strings, you may find [`str::as_bytes`] and [`String::into_bytes`] useful.
    pub fn add<'a, T>(&mut self, value: T) -> usize where T: Into<Cow<'a, [u8]>> {
        self.add_internal(value.into())
    }

    /// Removes all content from the pool.
    /// This is the only non-append-only operation, and is just meant to support resource reuse.
    pub fn clear(&mut self) {
        self.data.clear();
        self.slices.clear();
    }
    /// Gets the number of (distinct) slices contained in the pool.
    pub fn len(&self) -> usize {
        self.slices.len()
    }
    /// Checks if the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.slices.is_empty()
    }

    /// Iterates over all (distinct) slices contained in the pool in insertion order.
    pub fn iter(&self) -> Iter {
        Iter { data: &self.data, iter: self.slices.iter() }
    }
    /// Gets the slice at the specified index, or [`None`] if `index` is not a valid slice index returned by [`BinPool::add`].
    pub fn get(&self, index: usize) -> Option<&[u8]> {
        self.slices.get(index).map(|s| &self.data[s.src][s.start..s.start+s.length])
    }

    /// Gets the total number of bytes from (distinct) slices that were added to the pool.
    /// Note that the space needed to represent these slices may be significantly smaller (see [`BinPool::backing_bytes`]).
    pub fn bytes(&self) -> usize {
        self.slices.iter().fold(0, |v, s| v + s.length)
    }
    /// Gets the total number of bytes backing the stored slices.
    pub fn backing_bytes(&self) -> usize {
        self.data.iter().fold(0, |v, s| v + s.len())
    }

    /// Gets a reference to the backing data.
    pub fn as_backing(&self) -> (&Vec<Vec<u8>>, &Vec<SliceInfo>) {
        (&self.data, &self.slices)
    }
    /// Gets the backing data.
    pub fn into_backing(self) -> (Vec<Vec<u8>>, Vec<SliceInfo>) {
        (self.data, self.slices)
    }
}

/// Iterates over the (distinct) slices of a [`BinPool`] in insertion order.
pub struct Iter<'a> {
    data: &'a [Vec<u8>],
    iter: slice::Iter<'a, SliceInfo>,
}
impl<'a> Iterator for Iter<'a> {
    type Item = &'a [u8];
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|s| &self.data[s.src][s.start..s.start+s.length])
    }
}
impl<'a> FusedIterator for Iter<'a> {}

#[test]
fn test_binary_pool() {
    let mut s = BinPool::new();
    let checked_add = |s: &mut BinPool, v: Vec<u8>| {
        let res = s.add(&v);
        assert_eq!(s.get(res).unwrap(), &v);
        res
    };

    assert_eq!(s.len(), 0);
    assert!(s.is_empty());
    assert_eq!(s.iter().count(), 0);
    assert_eq!(s.bytes(), 0);
    assert_eq!(s.backing_bytes(), 0);

    assert_eq!(checked_add(&mut s, vec![1, 2, 3]), 0);
    assert_eq!(s.len(), 1);
    assert!(!s.is_empty());
    assert_eq!(s.iter().collect::<Vec<_>>(), vec![[1, 2, 3].as_ref()]);
    assert_eq!(s.data, &[[1, 2, 3].as_ref()]);
    assert_eq!(s.bytes(), 3);
    assert_eq!(s.backing_bytes(), 3);

    assert_eq!(checked_add(&mut s, vec![2, 3]), 1);
    assert_eq!(s.len(), 2);
    assert_eq!(s.iter().collect::<Vec<_>>(), vec![[1, 2, 3].as_ref(), [2, 3].as_ref()]);
    assert_eq!(s.data, &[[1, 2, 3].as_ref()]);
    assert_eq!(s.bytes(), 5);
    assert_eq!(s.backing_bytes(), 3);

    assert_eq!(checked_add(&mut s, vec![2, 3]), 1);
    assert_eq!(s.len(), 2);
    assert_eq!(s.iter().collect::<Vec<_>>(), vec![[1, 2, 3].as_ref(), [2, 3].as_ref()]);
    assert_eq!(s.data, &[[1, 2, 3].as_ref()]);
    assert_eq!(s.bytes(), 5);
    assert_eq!(s.backing_bytes(), 3);

    assert_eq!(checked_add(&mut s, vec![1, 2, 3]), 0);
    assert_eq!(s.len(), 2);
    assert_eq!(s.iter().collect::<Vec<_>>(), vec![[1, 2, 3].as_ref(), [2, 3].as_ref()]);
    assert_eq!(s.data, &[[1, 2, 3].as_ref()]);
    assert_eq!(s.bytes(), 5);
    assert_eq!(s.backing_bytes(), 3);

    assert_eq!(checked_add(&mut s, vec![2, 3, 4, 5]), 2);
    assert_eq!(s.len(), 3);
    assert_eq!(s.iter().collect::<Vec<_>>(), vec![[1, 2, 3].as_ref(), [2, 3].as_ref(), [2, 3, 4, 5].as_ref()]);
    assert_eq!(s.data, &[[1, 2, 3].as_ref(), [2, 3, 4, 5].as_ref()]);
    assert_eq!(s.bytes(), 9);
    assert_eq!(s.backing_bytes(), 7);

    assert_eq!(checked_add(&mut s, vec![2, 3, 4, 5]), 2);
    assert_eq!(s.len(), 3);
    assert_eq!(s.iter().collect::<Vec<_>>(), vec![[1, 2, 3].as_ref(), [2, 3].as_ref(), [2, 3, 4, 5].as_ref()]);
    assert_eq!(s.data, &[[1, 2, 3].as_ref(), [2, 3, 4, 5].as_ref()]);
    assert_eq!(s.bytes(), 9);
    assert_eq!(s.backing_bytes(), 7);

    assert_eq!(checked_add(&mut s, vec![1, 2, 3, 4]), 3);
    assert_eq!(s.len(), 4);
    assert_eq!(s.iter().collect::<Vec<_>>(), vec![[1, 2, 3].as_ref(), [2, 3].as_ref(), [2, 3, 4, 5].as_ref(), [1, 2, 3, 4].as_ref()]);
    assert_eq!(s.data, &[[1, 2, 3, 4].as_ref(), [2, 3, 4, 5].as_ref()]);
    assert_eq!(s.bytes(), 13);
    assert_eq!(s.backing_bytes(), 8);

    {
        let mut s = s.clone();
        assert_eq!(checked_add(&mut s, vec![255, 69, 1, 2, 3, 4, 5, 0, 0, 10, 20]), 4);
        assert_eq!(s.len(), 5);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![[1, 2, 3].as_ref(), [2, 3].as_ref(), [2, 3, 4, 5].as_ref(), [1, 2, 3, 4].as_ref(),
            [255, 69, 1, 2, 3, 4, 5, 0, 0, 10, 20].as_ref()]);
        assert_eq!(s.data, &[[255, 69, 1, 2, 3, 4, 5, 0, 0, 10, 20].as_ref()]);
        assert_eq!(s.bytes(), 24);
        assert_eq!(s.backing_bytes(), 11);
    }
    {
        let mut s = s.clone();
        assert_eq!(checked_add(&mut s, vec![2, 3, 4, 5, 0, 0, 10, 20]), 4);
        assert_eq!(s.len(), 5);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![[1, 2, 3].as_ref(), [2, 3].as_ref(), [2, 3, 4, 5].as_ref(), [1, 2, 3, 4].as_ref(),
            [2, 3, 4, 5, 0, 0, 10, 20].as_ref()]);
            assert_eq!(s.data, &[[1, 2, 3, 4].as_ref(), [2, 3, 4, 5, 0, 0, 10, 20].as_ref()]);
            assert_eq!(s.bytes(), 21);
            assert_eq!(s.backing_bytes(), 12);
    }
    {
        let mut s = s.clone();
        assert_eq!(checked_add(&mut s, vec![255, 69, 1, 2, 3, 4]), 4);
        assert_eq!(s.len(), 5);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![[1, 2, 3].as_ref(), [2, 3].as_ref(), [2, 3, 4, 5].as_ref(), [1, 2, 3, 4].as_ref(),
            [255, 69, 1, 2, 3, 4].as_ref()]);
        assert_eq!(s.data, &[[255, 69, 1, 2, 3, 4].as_ref(), [2, 3, 4, 5].as_ref()]);
        assert_eq!(s.bytes(), 19);
        assert_eq!(s.backing_bytes(), 10);
    }

    s.clear();
    assert_eq!(s.len(), 0);
    assert!(s.is_empty());
    assert_eq!(s.iter().count(), 0);
    assert_eq!(s.bytes(), 0);
    assert_eq!(s.backing_bytes(), 0);

    assert_eq!(checked_add(&mut s, vec![6, 6, 6]), 0);
    assert_eq!(s.len(), 1);
    assert!(!s.is_empty());
    assert_eq!(s.iter().collect::<Vec<_>>(), vec![[6, 6, 6].as_ref()]);
    assert_eq!(s.data, &[[6, 6, 6].as_ref()]);
    assert_eq!(s.bytes(), 3);
    assert_eq!(s.backing_bytes(), 3);

    assert_eq!(checked_add(&mut s, vec![2, 3, 4]), 1);
    assert_eq!(s.len(), 2);
    assert_eq!(s.iter().collect::<Vec<_>>(), vec![[6, 6, 6].as_ref(), [2, 3, 4].as_ref()]);
    assert_eq!(s.data, &[[6, 6, 6].as_ref(), [2, 3, 4].as_ref()]);
    assert_eq!(s.bytes(), 6);
    assert_eq!(s.backing_bytes(), 6);

    assert_eq!(checked_add(&mut s, vec![2, 3]), 2);
    assert_eq!(s.len(), 3);
    assert_eq!(s.iter().collect::<Vec<_>>(), vec![[6, 6, 6].as_ref(), [2, 3, 4].as_ref(), [2, 3].as_ref()]);
    assert_eq!(s.data, &[[6, 6, 6].as_ref(), [2, 3, 4].as_ref()]);
    assert_eq!(s.bytes(), 8);
    assert_eq!(s.backing_bytes(), 6);

    assert_eq!(checked_add(&mut s, vec![1, 2, 3, 6, 6, 6]), 3);
    assert_eq!(s.len(), 4);
    assert_eq!(s.iter().collect::<Vec<_>>(), vec![[6, 6, 6].as_ref(), [2, 3, 4].as_ref(), [2, 3].as_ref(), [1, 2, 3, 6, 6, 6].as_ref()]);
    assert_eq!(s.data, &[[1, 2, 3, 6, 6, 6].as_ref(), [2, 3, 4].as_ref()]);
    assert_eq!(s.bytes(), 14);
    assert_eq!(s.backing_bytes(), 9);

    assert_eq!(checked_add(&mut s, vec![2, 3]), 2);
    assert_eq!(s.len(), 4);
    assert_eq!(s.iter().collect::<Vec<_>>(), vec![[6, 6, 6].as_ref(), [2, 3, 4].as_ref(), [2, 3].as_ref(), [1, 2, 3, 6, 6, 6].as_ref()]);
    assert_eq!(s.data, &[[1, 2, 3, 6, 6, 6].as_ref(), [2, 3, 4].as_ref()]);
    assert_eq!(s.bytes(), 14);
    assert_eq!(s.backing_bytes(), 9);

    {
        let mut s = BinPool::new();
        assert_eq!(checked_add(&mut s, vec![0]), 0);
        assert_eq!(checked_add(&mut s, vec![1]), 1);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![[0].as_ref(), [1].as_ref()]);
        assert_eq!(s.data, vec![[0].as_ref(), [1].as_ref()]);
        assert_eq!(checked_add(&mut s, vec![0, 1]), 2);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![[0].as_ref(), [1].as_ref(), [0, 1].as_ref()]);
        assert_eq!(s.data, vec![[0, 1].as_ref()]);
    }
    {
        let mut s = BinPool::new();
        assert_eq!(checked_add(&mut s, vec![0]), 0);
        assert_eq!(checked_add(&mut s, vec![1]), 1);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![[0].as_ref(), [1].as_ref()]);
        assert_eq!(s.data, vec![[0].as_ref(), [1].as_ref()]);
        assert_eq!(checked_add(&mut s, vec![1, 0]), 2);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![[0].as_ref(), [1].as_ref(), [1, 0].as_ref()]);
        assert_eq!(s.data, vec![[1, 0].as_ref()]);
    }
    {
        let mut s = BinPool::new();
        assert_eq!(checked_add(&mut s, vec![]), 0);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![[].as_ref()]);
        assert_eq!(s.data, vec![[].as_ref()]);
        assert_eq!(checked_add(&mut s, vec![5]), 1);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![[].as_ref(), [5].as_ref()]);
        assert_eq!(s.data, vec![[5].as_ref()]);
    }
    {
        let mut s = BinPool::new();
        assert_eq!(checked_add(&mut s, vec![7]), 0);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![[7].as_ref()]);
        assert_eq!(s.data, vec![[7].as_ref()]);
        assert_eq!(checked_add(&mut s, vec![]), 1);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![[7].as_ref(), [].as_ref()]);
        assert_eq!(s.data, vec![[7].as_ref()]);
    }
}
