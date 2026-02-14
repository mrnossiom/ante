//! Various methods for validating the well-formedness of [Mir]
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rustc_hash::FxHashSet;

use crate::mir::{Definition, Instruction, Mir, Value};

impl Mir {
    /// Ensures:
    /// - Each referenced [DefinitionId] corresponds to a [Definition] or extern item in this Mir
    pub(crate) fn assert_fully_linked(self) -> Self {
        self.definitions.par_iter().for_each(|(_, definition)| {
            definition.assert_fully_linked(&self);
        });
        self
    }
}

impl Definition {
    fn assert_fully_linked(&self, mir: &Mir) {
        let mut referenced_ids = FxHashSet::default();
        referenced_ids.insert(self.id);

        for instruction in self.instructions.values() {
            instruction.for_each_value(|value| {
                if let Value::Definition(definition_id) = value {
                    referenced_ids.insert(*definition_id);
                }
            });

            if let Instruction::Instantiate(id, _) = instruction {
                referenced_ids.insert(*id);
            }
        }

        for block in self.blocks.values() {
            block.terminator.as_ref().unwrap().for_each_value(|value| {
                if let Value::Definition(definition_id) = value {
                    referenced_ids.insert(*definition_id);
                }
            });
        }

        for id in referenced_ids {
            if !mir.definitions.contains_key(&id) && !mir.external.contains_key(&id) {
                eprintln!("No definition for id {id:?}");
            }
        }
    }
}
