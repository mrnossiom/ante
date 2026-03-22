//! A MIR pass to replace unnecessary closures with free functions.
//!
//! Since the MIR builder checks for this as well, this pass should only
//! be run after monomorphization specializes some closure environments
//! into [Type::NoClosureEnv]
use rustc_hash::FxHashMap;

use crate::mir::{BlockId, Definition, DefinitionId, Instruction, Mir, Type, Value};

impl Mir {
    pub fn closure_deconvert(mut self) -> Self {
        let definition_types = self.externals.iter()
            .map(|(id, external)| (*id, external.typ.clone()))
            .chain(self.definitions.iter().map(|(id, def)| (*id, def.typ.clone())))
            .collect();

        for definition in self.definitions.values_mut() {
            closure_deconvert(definition, &definition_types);
        }
        self
    }
}

fn closure_deconvert(definition: &mut Definition, definition_types: &FxHashMap<DefinitionId, Type>) {
    // If this is a closure whose environment has been specialized into an empty environment,
    // remove the environment parameter.
    remove_no_closure_env_parameters(definition);

    for (instruction_id, instruction) in definition.instructions.iter_mut() {
        match instruction {
            // If `function` does not need to be a closure, turn it into a free function call
            Instruction::CallClosure { closure, arguments } => {
                let is_closure = match closure {
                    Value::InstructionResult(id) => definition.instruction_result_types[*id].is_closure(),
                    Value::Parameter(block_id, idx) => {
                        definition.blocks[*block_id].parameter_types[*idx as usize].is_closure()
                    },
                    Value::Definition(id) => definition_types[id].is_closure(),
                    _ => true,
                };
                if !is_closure {
                    let function = *closure;
                    let args = std::mem::take(arguments);
                    *instruction = Instruction::Call { function, arguments: args };
                }
            },

            // If `function` does not need to be a closure, turn it into a free function
            Instruction::PackClosure { function, .. } => {
                let result_type = &definition.instruction_result_types[instruction_id];
                if !result_type.is_closure() {
                    *instruction = Instruction::Id(*function);
                }
            },
            _ => (),
        }
    }
}

fn remove_no_closure_env_parameters(definition: &mut Definition) {
    let mut to_remove = Vec::new();
    for (i, parameter_type) in definition.entry_block().parameter_types.iter().enumerate() {
        if *parameter_type == Type::NO_CLOSURE_ENV {
            to_remove.push(Value::Parameter(BlockId::ENTRY_BLOCK, i as u32));
        }
    }

    if to_remove.is_empty() {
        return;
    }

    for instruction in definition.instructions.values_mut() {
        match instruction {
            Instruction::Call { function: _, arguments: values }
            | Instruction::CallClosure { closure: _, arguments: values }
            | Instruction::MakeTuple(values) => {
                values.retain(|value| !to_remove.contains(value));
            },
            // Instruction::PackClosure { function, environment } => {
            //     if to_remove.contains(environment) {
            //         *instruction = Instruction::Id(*function);
            //     }
            // },
            // Instruction::StackAlloc(value) => todo!(),
            // Instruction::Transmute(value) => todo!(),
            // Instruction::SizeOf(value) => todo!(),
            // Other instructions cannot contain values of type `NoClosureEnv`
            _ => (),
        }
    }
}
