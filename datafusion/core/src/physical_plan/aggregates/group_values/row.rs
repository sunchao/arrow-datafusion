// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use crate::physical_plan::aggregates::group_values::GroupValues;
use ahash::RandomState;
use arrow::row::{RowConverter, Rows, SortField};
use arrow_array::ArrayRef;
use arrow_schema::SchemaRef;
use datafusion_common::Result;
use datafusion_execution::memory_pool::proxy::VecAllocExt;
use datafusion_physical_expr::hash_utils::create_hashes;
use datafusion_physical_expr::EmitTo;

/// A [`GroupValues`] making use of [`Rows`]
pub struct GroupValuesRows {
    /// Converter for the group values
    row_converter: RowConverter,

    /// Logically maps group values to a group_index in
    /// [`Self::group_values`] and in each accumulator
    ///
    /// keys: u64 hashes of the GroupValue
    /// values: (hash, group_index)
    hash_table: Vec<usize>,

    /// The hashes for all the group by values
    hashes: Vec<u64>,

    /// The current capacity of [`Self.map`]
    capacity: usize,

    /// `group_values[i]` holds the group value for group_index `i`.
    ///
    /// The row format is used to compare group keys quickly and store
    /// them efficiently in memory. Quick comparison is especially
    /// important for multi-column group keys.
    ///
    /// [`Row`]: arrow::row::Row
    group_values: Rows,

    /// buffer to be reused to store hashes
    current_hashes: Vec<u64>,

    /// Whether `emit` has been called in this instance. This is used to check
    /// that `intern` shall not be called after `emit`.
    has_emitted: bool,

    // Stores the offset of each row in the [`Self.map`].
    //
    // This is getting updated as we keep iterating on each input batch. It is
    // reused across input batches.
    current_offsets: Vec<usize>,

    // Stores the indexes in the current input rows for which we should create
    // new entries in this [`GroupValues`].
    new_entries: Vec<usize>,

    // Stores the indices in the current input rows for which we should perform
    // equality check against the rows in [`Self::group_values`]
    need_equality_check: Vec<usize>,

    // Stores the indices in the current input rows for which both hash and
    // value equality check failed. We'll need to do probing on these rows and
    // perform the check in a new iteration.
    no_match: Vec<usize>,

    /// Random state for creating hashes
    random_state: RandomState,
}

/// Load factor of this hash table, rehash will be triggered when this is reached.
const LOAD_FACTOR: f64 = 1.5;
const INITIAL_CAPACITY: usize = 8192;

impl GroupValues for GroupValuesRows {
    fn intern(&mut self, cols: &[ArrayRef], groups: &mut Vec<usize>) -> Result<()> {
        if self.has_emitted {
            return Err(datafusion_common::DataFusionError::Internal(
                "intern() should not be called after emit()!".to_string(),
            ));
        }

        // Convert the group keys into the row format
        // Avoid reallocation when https://github.com/apache/arrow-rs/issues/4479 is available
        let group_rows = self.row_converter.convert_columns(cols)?;
        let n_rows = group_rows.num_rows();

        // tracks to which group each of the input rows belongs
        groups.clear();

        // 1.1 Calculate the group keys for the group values
        let batch_hashes = &mut self.current_hashes;
        batch_hashes.clear();
        batch_hashes.resize(n_rows, 0);
        create_hashes(cols, &self.random_state, batch_hashes)?;

        // rehash if necessary
        let current_n_rows = self.hashes.len();
        if current_n_rows + n_rows > self.resize_threshold() {
            self.rehash(current_n_rows + n_rows);
        }

        debug_assert!(is_power_of_2(self.capacity));
        let bit_mask = self.capacity - 1;

        // first, initialize the hash table offsets for the input
        self.current_offsets.resize(n_rows, 0);
        for row_idx in 0..n_rows {
            let hash = self.current_hashes[row_idx];
            let hash_table_idx = (hash as usize) & bit_mask;
            self.current_offsets[row_idx] = hash_table_idx;
        }

        // initially, `selection_vector[i]` = row at `i`
        let mut selection_vector: Vec<usize> = (0..n_rows).collect();
        let mut remaining_entries = selection_vector.len();

        // reserve beforehand, so we can operate on the raw pointer later, without worring about
        // resizing.
        // self.hashes.reserve(n_rows);

        // reserve enough space for the auxillary data structures
        self.new_entries.resize(n_rows, 0);
        self.need_equality_check.resize(n_rows, 0);
        self.no_match.resize(n_rows, 0);

        let mut num_iter = 1;

        // Repeatedly process the current `selection_vector` and put the rows into 3 different
        // vectors:
        //   - `new_entries`: new rows in the hash table
        //   - `need_equality_check`: the hash table is already occupied with the same hash value
        //   - `no_match`: the hash table is occupied and hash value is differnt
        //
        // For `new_entries`, we'll use a loop to create new entries in the corresponding hash
        // table slots.
        //
        // For each element in the `need_equality_check`, we'll use a loop to perform
        // equality check. If the check succeeds, nothing needs to be done since we've found the
        // hash table entry that's already created for the row. Otherwise, it means we need to do
        // further probing and thus the row is added to `no_match`.
        //
        // For the elements in `no_match`, we'll use a loop to do linear probing. They become the
        // `selection_vector` for the next iteration.
        while remaining_entries > 0 {
            let mut n_new_entries = 0;
            let mut n_need_equality_check = 0;
            let mut n_no_match = 0;

            selection_vector
                .iter()
                .take(remaining_entries)
                .for_each(|row_idx| {
                    let row_idx = *row_idx;
                    let hash = self.current_hashes[row_idx];
                    let ht_offset = self.current_offsets[row_idx];
                    let offset = self.hash_table[ht_offset];

                    if offset == 0 {
                        // the slot is empty, so we can create a new entry here
                        self.new_entries[n_new_entries] = row_idx;
                        n_new_entries += 1;

                        // we increment the slot entry offset by 1 to reserve the special value
                        // 0 for the scenario when the slot in the
                        // hash table is unoccupied.
                        self.hash_table[ht_offset] = self.hashes.len() + 1;
                        // also update hash for this slot so it can be used later
                        self.hashes.push(hash);
                    } else if self.hashes[offset as usize - 1] == hash {
                        // slot is not empty, and hash value match, now need to do equality
                        // check
                        self.need_equality_check[n_need_equality_check] = row_idx;
                        n_need_equality_check += 1;
                    } else {
                        // slot is not empty, and hash value doesn't match, we have a hash
                        // collision and need to do probing
                        self.no_match[n_no_match] = row_idx;
                        n_no_match += 1;
                    }
                });

            self.process_new_entries(n_new_entries, &group_rows);

            self.process_need_equality_check(
                &group_rows,
                n_need_equality_check,
                &mut n_no_match,
            );

            // now we need to probing for those rows in `no_match`
            self.process_no_match(n_no_match, num_iter);

            std::mem::swap(&mut self.no_match, &mut selection_vector);
            remaining_entries = n_no_match;

            num_iter += 1;
        }

        self.current_offsets
            .iter()
            .take(n_rows)
            .for_each(|hash_table_offset| {
                groups.push(self.hash_table[*hash_table_offset] - 1);
            });

        Ok(())
    }

    fn size(&self) -> usize {
        self.row_converter.size()
            + self.group_values.size()
            + self.hash_table.allocated_size()
            + self.current_hashes.allocated_size()
            + self.current_offsets.allocated_size()
            + self.need_equality_check.allocated_size()
            + self.new_entries.allocated_size()
            + self.no_match.allocated_size()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize {
        self.hashes.len()
    }

    fn emit(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let result = match emit_to {
            EmitTo::All => {
                // Eventually we may also want to clear the hash table here
                self.row_converter.convert_rows(&self.group_values)?
            }
            EmitTo::First(n) => {
                let groups_rows = self.group_values.iter().take(n);
                let output = self.row_converter.convert_rows(groups_rows)?;
                // Clear out first n group keys by copying them to a new Rows.
                // TODO file some ticket in arrow-rs to make this more efficent?
                let mut new_group_values = self.row_converter.empty_rows(0, 0);
                for row in self.group_values.iter().skip(n) {
                    new_group_values.push(row);
                }
                std::mem::swap(&mut new_group_values, &mut self.group_values);

                // Removes the first n elements from the hashes
                self.hashes.drain(0..n);

                // Also update hash table and clear out emitted entries
                for i in 0..self.hash_table.len() {
                    let ht_offset = self.hash_table[i];
                    if ht_offset > 0 {
                        match ht_offset.checked_sub(n + 1) {
                            Some(sub) => self.hash_table[i] = sub + 1,
                            None => self.hash_table[i] = 0,
                        }
                    }
                }

                output
            }
        };

        self.has_emitted = true;
        Ok(result)
    }
}

impl GroupValuesRows {
    pub fn try_new(schema: SchemaRef) -> Result<Self> {
        let row_converter = RowConverter::new(
            schema
                .fields()
                .iter()
                .map(|f| SortField::new(f.data_type().clone()))
                .collect(),
        )?;

        let group_values = row_converter.empty_rows(0, 0);

        Ok(Self {
            row_converter,
            hash_table: vec![0; INITIAL_CAPACITY],
            capacity: INITIAL_CAPACITY,
            hashes: Vec::with_capacity(INITIAL_CAPACITY),
            group_values,
            has_emitted: false,
            current_hashes: Default::default(),
            random_state: Default::default(),
            current_offsets: Default::default(),
            new_entries: Default::default(),
            need_equality_check: Default::default(),
            no_match: Default::default(),
        })
    }

    /// The threshold for hash table resize.
    fn resize_threshold(&self) -> usize {
        (self.hash_table.capacity() as f64 / LOAD_FACTOR) as usize
    }

    fn process_new_entries(&mut self, num_new_entries: usize, group_rows: &Rows) {
        self.new_entries
            .iter()
            .take(num_new_entries)
            .for_each(|row_idx| {
                let row = group_rows.row(*row_idx);
                self.group_values.push(row);
            });
        assert_eq!(self.hashes.len(), self.group_values.num_rows());
    }

    fn process_need_equality_check(
        &mut self,
        group_rows: &Rows,
        n_need_equality_check: usize,
        n_no_match: &mut usize,
    ) {
        self.need_equality_check
            .iter()
            .take(n_need_equality_check)
            .for_each(|row_idx| {
                let row_idx = *row_idx;
                let ht_offset = self.current_offsets[row_idx];
                let offset = self.hash_table[ht_offset];

                let existing = self.group_values.row(offset - 1);
                let incoming = group_rows.row(row_idx);

                if existing != incoming {
                    self.no_match[*n_no_match] = row_idx;
                    *n_no_match += 1;
                }
            });
    }

    fn process_no_match(&mut self, num_no_match: usize, num_iter: usize) {
        // For the rows that have no match in the hash table, do quadratic probing
        // and try them in the next iteration.
        let delta = num_iter * num_iter;
        let bit_mask = self.capacity - 1;
        for i in 0..num_no_match {
            let row_idx = self.no_match[i];
            let slot_idx = self.current_offsets[row_idx] + delta;
            self.current_offsets[row_idx] = slot_idx & bit_mask;
        }
    }

    fn rehash(&mut self, new_capacity: usize) {
        let new_capacity = std::cmp::max(new_capacity, 2 * self.capacity);
        let new_capacity = new_capacity.next_power_of_two();
        assert!(is_power_of_2(new_capacity));

        let table_ptr = self.hash_table.as_ptr();
        let hashes_ptr = self.hashes.as_ptr();
        let mut new_table = vec![0; new_capacity];
        let new_table_ptr = new_table.as_mut_ptr();
        let new_bit_mask = new_capacity - 1;

        unsafe {
            for i in 0..self.capacity {
                let offset = *table_ptr.add(i);
                if offset != 0 {
                    let hash = *hashes_ptr.add(offset as usize - 1);

                    let mut new_idx = hash as usize & new_bit_mask;
                    let mut num_iter = 0;
                    while *new_table_ptr.add(new_idx) != 0 {
                        num_iter += 1;
                        new_idx += num_iter * num_iter;
                        new_idx &= new_bit_mask;
                    }
                    *new_table_ptr.add(new_idx) = offset;
                }
            }
        }

        self.hash_table = new_table;
        self.capacity = new_capacity;
    }
}

/// Checks whether the input `num` is power of 2.
#[inline]
pub fn is_power_of_2(num: usize) -> bool {
    num > 0 && num & (num - 1) == 0
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::ops::Range;
    use std::sync::Arc;

    use arrow_array::Int64Array;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;

    use super::*;

    #[test]
    fn test_basic() {
        let mut hash_table = get_hash_table();
        let mut out_indices = Vec::new();

        assert_eq!(hash_table.len(), 0);
        // assert_eq!(hash_table.capacity, TEST_BATCH_SIZE * 2);

        let total = 50;
        let inputs = get_input(0..total as i64);
        let res = hash_table.intern(&inputs, &mut out_indices);
        assert!(res.is_ok());

        // test that the number of entries in the hash table matches the number of input group
        // keys, and also that each group key occupies a separate slot in the table.
        let mut set = HashSet::new();
        let mut count = 0;
        let internal = hash_table.hash_table;
        for i in 0..hash_table.capacity {
            if internal[i] > 0 {
                count += 1;
                set.insert(internal[i]);
            }
        }

        assert_eq!(count, total);
        assert_eq!(set.len(), total);
    }

    #[test]
    fn test_rehash() {
        let mut hash_table = get_hash_table();
        let mut out_indices = Vec::new();

        assert_eq!(hash_table.len(), 0);
        //
        // add total of 300 rows, which should trigger the rehash and double its capacity
        let total: usize = 100;
        let inputs_1 = get_input(0..total as i64);
        let inputs_2 = get_input(total as i64..2 * total as i64);
        let inputs_3 = get_input(2 * total as i64..3 * total as i64);
        let _ = hash_table.intern(&inputs_1, &mut out_indices);
        let _ = hash_table.intern(&inputs_2, &mut out_indices);
        let _ = hash_table.intern(&inputs_3, &mut out_indices);
    }

    // Create a new [`GroupValuesRows`] for testing
    fn get_hash_table() -> GroupValuesRows {
        let schema = Schema::new(vec![Field::new("f", DataType::Int64, false)]);
        GroupValuesRows::try_new(Arc::new(schema)).unwrap()
    }

    // Generate `num_of_vectors` test Arrow arrays with longs, each containing numbers from 0 to
    // `num`.
    fn get_input(range: Range<i64>) -> Vec<ArrayRef> {
        let mut result = Vec::new();
        let data: Vec<i64> = range.collect();
        let v: ArrayRef = Arc::new(Int64Array::from(data));
        result.push(v);
        result
    }
}
