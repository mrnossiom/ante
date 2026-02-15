//! A pass separate to but related to monomorphization.
//!
//! Traverses the Mir replacing each union type with the largest variant in the union,
//! according to the current target machine.

use crate::mir::Mir;

impl Mir {
    /// Replace each union type used with the largest variant of that type.
    fn select_largest_variants(self) -> Self {
        todo!()
    }
}
