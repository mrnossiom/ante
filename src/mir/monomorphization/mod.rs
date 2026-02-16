//! This file contains the logic for specializing generics out of the MIR. This process is called
//! monomorphization and in Ante it is a Mir -> Mir transformation.
//!
//! The monomorphizer starts from the entry point to the program and from there builds a queue
//! of functions which need to be monomorphized. This queue can be processed concurrently with
//! each individual function being handled by a single [FunctionContext] object.
use std::{collections::hash_map::Entry, sync::Arc};

use dashmap::DashMap;
use inc_complete::DbGet;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rustc_hash::{FxHashMap, FxHashSet};

mod select_largest_variant;

use crate::{
    definition_collection::collect_all_items,
    incremental::{GetCrateGraph, GetItem, GetItemRaw, Parse, TargetPointerSize, TypeCheck},
    mir::{
        self, Definition, DefinitionId, GenericBindings, Instruction, Mir, Type, Value,
        builder::build_initial_mir_with_shared_map, next_definition_id,
    },
    parser::cst::Name,
};

/// Monomorphize the whole program, returning a MIR function if the item refers to a function.
/// If the item does not refer to a function (e.g. it is a type definition), `None` is returned.
///
/// Note that monomorphize needs access to every item to monomorphize at once - it may not be
/// called separately and combined via [Mir::extend] later as this will lead to missing generic
/// definitions which were not monomorphized. `items` must contain every item in the program.
pub(crate) fn monomorphize<Db>(compiler: &Db) -> Mir
where
    Db: DbGet<TypeCheck> + DbGet<GetItem> + DbGet<GetItemRaw> + DbGet<GetCrateGraph> + DbGet<Parse> + DbGet<TargetPointerSize> + Sync,
{
    let initial_mir = collect_all_items(compiler)
        .into_par_iter()
        .flat_map(|item| build_initial_mir_with_shared_map(compiler, item))
        .reduce(Mir::default, Mir::extend);

    let shared = SharedDefinitions::default();

    // If there are no generics this is an entry point to monomorphization.
    // If there are generics, then we'll either monomorphize this function later
    // when we find its type arguments, or never if it is unused.
    let monomorphic_definitions = initial_mir
        .definitions
        .iter()
        .filter(|(_, definition)| definition.is_monomorphic() || definition.name.as_str() == "main")
        .map(|(_, definition)| {
            shared.insert((definition.id, Arc::new(Vec::new())), definition.id);
            definition.clone()
        })
        .collect::<Vec<_>>();

    // TODO: Switch to a concurrent queue to better utilize each thread instead of having each
    // thread compile each function reachable from each monomorphic function.
    monomorphic_definitions
        .into_par_iter()
        .fold(Mir::default, |acc, definition| {
            let monomorphized = monomorphize_non_generic_definition(definition, &shared, &initial_mir);
            acc.extend(monomorphized)
        })
        .reduce(Mir::default, Mir::extend)
        .with_externals_and_names(initial_mir.external, initial_mir.names)
        .select_largest_variants(compiler)
        .assert_fully_linked()
        .assert_type_checks()
        .assert_no_unions_or_generics()
}

/// The entry point to monomorphization is any non-generic definition.
/// We can't start with generic definitions since they require type bindings from their callsite(s).
///
/// `initial_mir` is the Mir pre-monomorphization and is not modified.
fn monomorphize_non_generic_definition(
    definition: Definition, definitions: &SharedDefinitions, initial_mir: &Mir,
) -> Mir {
    let mut context = FunctionContext::new(definitions);
    context.monomorphize_definition(definition, initial_mir);

    while let Some(item) = context.queue.pop() {
        let Some(original_definition) = initial_mir.get(item.old_id) else {
            eprintln!(
                "Monomorphization: no definition for id {}, was monomorphize not given every top-level-item in a single invocation?",
                item.old_id
            );
            continue;
        };

        // If the original definition is already monomorphic, it will be covered by a different
        // call to [monomorphize_non_generic_definition].
        if !original_definition.is_monomorphic() {
            let mut definition = original_definition.clone_with_id(item.new_id);
            definition.generic_count = 0;
            definition.typ = item.monomorphized_type;
            context.generic_mapping = item.bindings.clone();
            context.monomorphize_definition(definition, initial_mir);
        }
    }

    Mir { definitions: context.finished_definitions, external: Default::default(), names: context.names }
}

struct FunctionContext<'local> {
    generic_mapping: Arc<GenericBindings>,

    queue: Vec<DefinitionToMonomorphize>,

    finished_definitions: FxHashMap<DefinitionId, Definition>,

    /// This is shared between all concurrent monomorphize calls
    definitions: &'local SharedDefinitions,

    names: FxHashMap<DefinitionId, Name>,
}

struct DefinitionToMonomorphize {
    /// The old id pre-monomorphization
    old_id: DefinitionId,
    /// The id referring to the monomorphized version of `old_id` with the given generic bindings
    new_id: DefinitionId,
    bindings: Arc<GenericBindings>,
    monomorphized_type: Type,
}

/// Maps (old_id, generic bindings) to a new [DefinitionId] referring to the newly monomorphized
/// version of `old_id` with the given generic type bindings.
type SharedDefinitions = DashMap<(DefinitionId, Arc<GenericBindings>), DefinitionId>;

impl<'local> FunctionContext<'local> {
    fn new(definitions: &'local SharedDefinitions) -> Self {
        Self {
            definitions,
            generic_mapping: Default::default(),
            queue: Default::default(),
            finished_definitions: Default::default(),
            names: Default::default(),
        }
    }

    fn monomorphize_definition(&mut self, mut definition: mir::Definition, initial_mir: &Mir) {
        if !self.generic_mapping.is_empty() {
            self.update_value_types(&mut definition);
        }

        let mut removed_ids = FxHashSet::default();

        // We can skip the blocks and go right to the instructions themselves. There shouldn't be
        // any that aren't used in a block.
        for (instruction_id, instruction) in definition.instructions.iter_mut() {
            if let Instruction::Instantiate(id, bindings) = instruction {
                let new_id = *self.definitions.entry((*id, bindings.clone())).or_insert_with(|| {
                    let new_id = next_definition_id();
                    removed_ids.insert(*id);
                    let typ = definition.instruction_result_types[instruction_id].clone();
                    self.names.insert(new_id, initial_mir.names[id].clone());
                    self.queue.push(DefinitionToMonomorphize {
                        old_id: *id,
                        new_id,
                        monomorphized_type: typ,
                        bindings: bindings.clone(),
                    });
                    new_id
                });

                if let Entry::Vacant(entry) = definition.definition_types.entry(new_id) {
                    let typ = definition.instruction_result_types[instruction_id].clone();
                    entry.insert(typ);
                }

                *instruction = Instruction::Id(Value::Definition(new_id));
            }
        }

        for id in removed_ids {
            definition.definition_types.remove(&id);
        }

        self.finished_definitions.insert(definition.id, definition);
    }

    fn update_value_types(&self, definition: &mut Definition) {
        for result_type in definition.instruction_result_types.values_mut() {
            self.specialize_type(result_type);
        }

        for definition_type in definition.definition_types.values_mut() {
            self.specialize_type(definition_type);
        }

        for block in definition.blocks.values_mut() {
            for parameter in block.parameter_types.iter_mut() {
                self.specialize_type(parameter);
            }
        }
    }

    /// Replace any instances of generics in `self.generic_mapping` of the given type with their mapping.
    /// The resulting type should be guaranteed free of [Type::Generic].
    fn specialize_type(&self, typ: &mut Type) {
        let recur = |typ| self.specialize_type(typ);

        match typ {
            Type::Primitive(_) => (),
            Type::Tuple(fields) => {
                let mut new_fields = fields.to_vec();
                new_fields.iter_mut().for_each(recur);
                *fields = Arc::new(new_fields);
            },
            Type::Function(function) => {
                let mut new_function = function.as_ref().clone();
                new_function.parameters.iter_mut().for_each(recur);
                recur(&mut new_function.return_type);
                *function = Arc::new(new_function);
            },
            Type::Union(variants) => {
                // TODO: Can we restructure the repr of Type so that this work is not repeated?
                let mut new_variants = variants.to_vec();
                new_variants.iter_mut().for_each(recur);
                *variants = Arc::new(new_variants);
            },
            Type::Generic(generic) => {
                let Some(mapping) = self.generic_mapping.get(generic.0 as usize) else {
                    unreachable!("Unmapped generic found in monomorphization: {generic:?}")
                };
                *typ = mapping.clone();
            },
        }
    }
}
