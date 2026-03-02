//! Various methods for validating the well-formedness of [Mir]
use rustc_hash::FxHashSet;

use crate::mir::{Definition, Instruction, InstructionId, Mir, PrimitiveType, TerminatorInstruction, Type, Value};

impl Mir {
    /// Ensures:
    /// - Each referenced [DefinitionId] corresponds to a [Definition] or extern item in this Mir
    pub(crate) fn assert_fully_linked(self) -> Self {
        //self.definitions.par_iter().for_each(|(_, definition)| {
        self.definitions.iter().for_each(|(_, definition)| {
            definition.assert_fully_linked(&self);
        });
        self
    }

    /// Asserts the types given to and returned from each instruction are valid
    pub(crate) fn assert_type_checks(self) -> Self {
        //self.definitions.par_iter().for_each(|(_, definition)| {
        self.definitions.iter().for_each(|(_, definition)| {
            definition.assert_type_checks(&self);
        });
        self
    }

    /// Union and generic types should be lowered into more explicit forms before handing off the
    /// Mir to the backend.
    pub(crate) fn assert_no_unions_or_generics(self) -> Self {
        //self.definitions.par_iter().for_each(|(_, definition)| {
        self.definitions.iter().for_each(|(_, definition)| {
            definition.assert_no_unions_or_generics(&self);
        });
        self
    }
}

macro_rules! instr_panic {
    ($this: expr, $instruction_id: expr, $mir: expr, $($msg: tt)*) => {{
        $this.annotate_error($instruction_id, $mir, &format!($($msg)*));
        panic!()
    }};
}

macro_rules! instr_assert {
    ($cond: expr, $this: expr, $instruction_id: expr, $mir: expr, $($msg: tt)* ) => {{
        if !$cond {
            $this.annotate_error($instruction_id, $mir, &format!($($msg)*));
            panic!()
        }
    }};
}

macro_rules! instr_assert_eq {
    ($lhs: expr, $rhs: expr, $this: expr, $instruction_id: expr, $mir: expr, $($msg: tt)*) => {{
        if $lhs != $rhs {
            $this.annotate_error($instruction_id, $mir, &format!($($msg)*));
            panic!()
        }
    }};
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
            if !mir.definitions.contains_key(&id) && !mir.externals.contains_key(&id) {
                panic!("Mir::assert_fully_linked: No definition for id {id:?}");
            }
        }
    }

    /// Asserts the argument & result types of each instruction are valid. If they are not, the Mir
    /// is not well-formed.
    fn assert_type_checks(&self, mir: &Mir) {
        self.type_check_instructions(mir);
        self.type_check_block_terminators(mir);
        self.assert_parameter_types_match_definition_type(mir);
    }

    // If this definition is a:
    // - Global: it should have no block parameters
    // - Function: `self.typ` should be a function type with parameters
    //   matching the entry block's parameters.
    fn assert_parameter_types_match_definition_type(&self, mir: &Mir) {
        let entry_block = self.entry_block();
        if self.is_global() {
            let parameters = entry_block.parameter_types.len();
            assert_eq!(parameters, 0, "\n{}\n\nGlobal should have 0 parameters", self.display(Some(mir)));
        } else {
            let parameters = match &self.typ {
                Type::Function(function_type) => &function_type.parameters,
                _ => panic!("\n{}\n\nFunction does not have a function type!", self.display(Some(mir))),
            };
            assert_eq!(
                *parameters,
                entry_block.parameter_types,
                "\n{}\n\nFunction parameters in type do not match entry block parameters",
                self.display(Some(mir))
            );
        }
    }

    // The macro calls here are too long so rustfmt puts every argument on a different line, ruining readability
    #[rustfmt::skip]
    fn type_check_instructions(&self, mir: &Mir) {
        for (id, instruction) in self.instructions.iter() {
            let result_type = &self.instruction_result_types[id];

            match instruction {
                Instruction::Call { function, arguments } => {
                    let function_type = mir.type_of_value(function, self);
                    let Type::Function(function_type) = function_type else {
                        instr_panic!(self, id, mir, "Called value is not a function, it is a(n) `{function_type}`")
                    };

                    instr_assert_eq!(function_type.parameters.len(), arguments.len(), self, id, mir, "parameter type len does not match arg type len");
                    for (i, (parameter, argument)) in function_type.parameters.iter().zip(arguments).enumerate() {
                        instr_assert_eq!(*parameter, mir.type_of_value(argument, self), self, id, mir, "Type mismatch in arg {i} of call");
                    }
                    instr_assert_eq!(function_type.return_type, *result_type, self, id, mir, "Function type result type does not match result type of call instruction");
                },
                Instruction::IndexTuple { tuple, index } => {
                    let tuple_type = mir.type_of_value(tuple, self);
                    let Type::Tuple(tuple_type) = tuple_type else {
                        instr_panic!(self, id, mir, "IndexTuple value is not a tuple, it is a(n) `{tuple_type}`")
                    };

                    instr_assert!((*index as usize) < tuple_type.len(), self, id, mir, "Index OOB");
                    instr_assert_eq!(tuple_type[*index as usize], *result_type, self, id, mir, "Element type from tuple != result type");
                },
                Instruction::MakeString(_) => {
                    instr_assert_eq!(*result_type, Type::string(), self, id, mir, "MakeString returns a non-string");
                },
                Instruction::MakeTuple(elements) => {
                    let Type::Tuple(tuple_type) = result_type else {
                        instr_panic!(self, id, mir, "MakeTuple result is not a tuple, it is a(n) `{result_type}`")
                    };
                    instr_assert_eq!(tuple_type.len(), elements.len(), self, id, mir, "Tuple type element length mismatch vs the actual elements length");
                    for (result, element) in tuple_type.iter().zip(elements) {
                        let element_type = mir.type_of_value(element, self);
                        instr_assert_eq!(*result, element_type, self, id, mir, "Tuple elem `{result}` != `{element_type}`");
                    }
                },
                Instruction::StackAlloc(_) => {
                    instr_assert_eq!(*result_type, Type::POINTER, self, id, mir, "Result type is not a pointer, it is `{result_type}`");
                },
                Instruction::Transmute(_) => (),
                Instruction::Instantiate(def_id, generic_args) => {
                    let target_type = &mir.type_of_value(&Value::Definition(*def_id), self).substitute(generic_args);
                    instr_assert_eq!(result_type, target_type, self, id, mir, "Result type `{result_type}` does not match manually substited type `{target_type}`");
                },
                Instruction::Id(value) => {
                    let value_type = mir.type_of_value(value, self);
                    instr_assert_eq!(*result_type, value_type, self, id, mir, "Value type `{value_type}` != result type `{result_type}`");
                },
                Instruction::AddInt(a, b)
                | Instruction::SubInt(a, b)
                | Instruction::MulInt(a, b)
                | Instruction::BitwiseAnd(a, b)
                | Instruction::BitwiseOr(a, b)
                | Instruction::BitwiseXor(a, b) => {
                    let a_type = mir.type_of_value(a, self);
                    let b_type = mir.type_of_value(b, self);
                    instr_assert!(a_type.is_int(), self, id, mir, "Argument type is not an integer");
                    instr_assert_eq!(a_type, b_type, self, id, mir, "Argument types do not match: {a_type} != {b_type}");
                    instr_assert_eq!(a_type, *result_type, self, id, mir, "Argument type does not match result type `{a_type}` != `{result_type}`");
                },

                Instruction::AddFloat(a, b)
                | Instruction::SubFloat(a, b)
                | Instruction::MulFloat(a, b)
                | Instruction::DivFloat(a, b)
                | Instruction::ModFloat(a, b) => {
                    let a_type = mir.type_of_value(a, self);
                    let b_type = mir.type_of_value(b, self);
                    instr_assert!(a_type.is_float(), self, id, mir, "Argument type is not a float");
                    instr_assert_eq!(a_type, b_type, self, id, mir, "Argument types do not match: {a_type} != {b_type}");
                    instr_assert_eq!(a_type, *result_type, self, id, mir, "Argument type does not match result type `{a_type}` != `{result_type}`");
                },

                Instruction::DivSigned(a, b) | Instruction::ModSigned(a, b) => {
                    let a_type = mir.type_of_value(a, self);
                    let b_type = mir.type_of_value(b, self);
                    instr_assert!(a_type.is_signed_int(), self, id, mir, "Argument type is not a signed int");
                    instr_assert_eq!(a_type, b_type, self, id, mir, "Argument types do not match: {a_type} != {b_type}");
                    instr_assert_eq!(a_type, *result_type, self, id, mir, "Argument type does not match result type `{a_type}` != `{result_type}`");
                },

                Instruction::DivUnsigned(a, b) | Instruction::ModUnsigned(a, b) => {
                    let a_type = mir.type_of_value(a, self);
                    let b_type = mir.type_of_value(b, self);
                    instr_assert!(a_type.is_unsigned_int(), self, id, mir, "Argument type is not an unsigned int");
                    instr_assert_eq!(a_type, b_type, self, id, mir, "Argument types do not match: {a_type} != {b_type}");
                    instr_assert_eq!(a_type, *result_type, self, id, mir, "Argument type does not match result type `{a_type}` != `{result_type}`");
                }

                Instruction::LessSigned(a, b) => {
                    let a_type = mir.type_of_value(a, self);
                    let b_type = mir.type_of_value(b, self);
                    instr_assert!(a_type.is_signed_int(), self, id, mir, "Argument type is not a signed int");
                    instr_assert_eq!(a_type, b_type, self, id, mir, "Argument types do not match: {a_type} != {b_type}");
                    instr_assert_eq!(*result_type, Type::BOOL, self, id, mir, "Result type `{result_type}` is not a Bool");
                },

                Instruction::LessUnsigned(a, b) => {
                    let a_type = mir.type_of_value(a, self);
                    let b_type = mir.type_of_value(b, self);
                    instr_assert!(a_type.is_unsigned_int(), self, id, mir, "Argument type is not an unsigned int");
                    instr_assert_eq!(a_type, b_type, self, id, mir, "Argument types do not match: {a_type} != {b_type}");
                    instr_assert_eq!(*result_type, Type::BOOL, self, id, mir, "Result type `{result_type}` is not a Bool");
                },

                Instruction::LessFloat(a, b) | Instruction::EqFloat(a, b) => {
                    let a_type = mir.type_of_value(a, self);
                    let b_type = mir.type_of_value(b, self);
                    instr_assert!(a_type.is_float(), self, id, mir, "Argument type is not a float");
                    instr_assert_eq!(a_type, b_type, self, id, mir, "Argument types do not match: {a_type} != {b_type}");
                    instr_assert_eq!(*result_type, Type::BOOL, self, id, mir, "Result type `{result_type}` is not a Bool");
                },

                Instruction::EqInt(a, b) => {
                    let a_type = mir.type_of_value(a, self);
                    let b_type = mir.type_of_value(b, self);
                    let valid = a_type.is_int() || a_type == Type::BOOL || a_type == Type::CHAR;
                    instr_assert!(valid, self, id, mir, "Argument type is not an integer, bool, or char");
                    instr_assert_eq!(a_type, b_type, self, id, mir, "Argument types do not match: {a_type} != {b_type}");
                    instr_assert_eq!(*result_type, Type::BOOL, self, id, mir, "Result type `{result_type}` is not a Bool");
                },

                Instruction::BitwiseNot(value) => {
                    let value_type = mir.type_of_value(value, self);
                    instr_assert!(value_type.is_int(), self, id, mir, "Argument type is not an integer");
                    instr_assert_eq!(value_type, *result_type, self, id, mir, "Argument type does not match result type `{value_type}` != `{result_type}`");
                },

                Instruction::SignExtend(value) => {
                    let value_type = mir.type_of_value(value, self);
                    instr_assert!(value_type.is_signed_int(), self, id, mir, "Argument type is not a signed integer");
                    instr_assert!(result_type.is_int(), self, id, mir, "Result type is not an integer");
                },
                Instruction::ZeroExtend(value) => {
                    let value_type = mir.type_of_value(value, self);
                    instr_assert!(value_type.is_unsigned_int() || value_type == Type::BOOL || value_type == Type::CHAR, self, id, mir, "Argument type is not an unsigned integer");
                    instr_assert!(result_type.is_int(), self, id, mir, "Result type is not an integer");
                },
                Instruction::SignedToFloat(value) => {
                    instr_assert!(mir.type_of_value(value, self).is_signed_int(), self, id, mir, "Argument type is not a signed integer");
                    instr_assert!(result_type.is_float(), self, id, mir, "Result type is not a float");
                },
                Instruction::UnsignedToFloat(value) => {
                    instr_assert!(mir.type_of_value(value, self).is_unsigned_int(), self, id, mir, "Argument type is not an unsigned integer");
                    instr_assert!(result_type.is_float(), self, id, mir, "Result type is not a float");
                },
                Instruction::FloatToSigned(value) => {
                    instr_assert!(mir.type_of_value(value, self).is_float(), self, id, mir, "Argument type is not a float");
                    instr_assert!(result_type.is_signed_int(), self, id, mir, "Result type is not a signed integer");
                },
                Instruction::FloatToUnsigned(value) => {
                    instr_assert!(mir.type_of_value(value, self).is_float(), self, id, mir, "Argument type is not a float");
                    instr_assert!(result_type.is_unsigned_int(), self, id, mir, "Result type is not an unsigned integer");
                },
                Instruction::FloatPromote(value) => {
                    instr_assert!(mir.type_of_value(value, self).is_float(), self, id, mir, "Argument type is not a float");
                    instr_assert!(result_type.is_float(), self, id, mir, "Result type is not a float");
                },
                Instruction::FloatDemote(value) => {
                    instr_assert!(mir.type_of_value(value, self).is_float(), self, id, mir, "Argument type is not a float");
                    instr_assert!(result_type.is_float(), self, id, mir, "Result type is not a float");
                },
                Instruction::Truncate(value) => {
                    let typ = mir.type_of_value(value, self);
                    instr_assert!(typ.can_be_used_as_integer(), self, id, mir, "Argument type is not an integer");
                    instr_assert!(result_type.can_be_used_as_integer(), self, id, mir, "Result type is not an integer");
                },
                Instruction::Deref(value) => {
                    let value_type = mir.type_of_value(value, self);
                    instr_assert!(matches!(value_type, Type::POINTER), self, id, mir, "Argument type is not a pointer");
                },
                Instruction::SizeOf(_) => (),
            }
        }
    }

    fn type_check_block_terminators(&self, mir: &Mir) {
        let block_arg_type_checks = |(target, arg): &(_, Option<Value>)| {
            let target_block = &self.blocks[*target];
            match arg {
                Some(arg) => {
                    assert_eq!(target_block.parameter_types.len(), 1);
                    assert_eq!(target_block.parameter_types[0], mir.type_of_value(arg, self));
                },
                None => {
                    assert_eq!(target_block.parameter_types.len(), 0);
                },
            }
        };

        for (block_id, block) in self.blocks.iter() {
            match block.terminator.as_ref() {
                Some(TerminatorInstruction::Jmp(target)) => block_arg_type_checks(target),
                Some(TerminatorInstruction::If { condition, then, else_, end: _ }) => {
                    assert_eq!(mir.type_of_value(condition, self), Type::BOOL);
                    block_arg_type_checks(then);
                    block_arg_type_checks(else_);
                },
                Some(TerminatorInstruction::Switch { int_value, cases, else_, end: _ }) => {
                    let int_type = mir.type_of_value(int_value, self);
                    assert!(matches!(int_type, Type::Primitive(PrimitiveType::Int(_))));

                    for case in cases {
                        block_arg_type_checks(case);
                    }

                    if let Some(else_) = else_ {
                        block_arg_type_checks(else_);
                    }
                },
                Some(TerminatorInstruction::Unreachable) => (),
                Some(TerminatorInstruction::Return(value)) => {
                    let return_type = match self.typ.function_return_type() {
                        Some(return_type) => return_type,
                        None => &self.typ,
                    };
                    assert_eq!(
                        mir.type_of_value(value, self),
                        *return_type,
                        "Returned value's type does not match function return type:\n{}",
                        self.display(Some(mir))
                    );
                },
                Some(TerminatorInstruction::Result(value)) => {
                    assert_eq!(
                        mir.type_of_value(value, self),
                        self.typ,
                        "Result value's type does not match the type of the global:\n{}",
                        self.display(Some(mir))
                    );
                },
                None => panic!("type_check_block_terminators: {block_id} has no terminators!"),
            }
        }
    }

    /// Helper to show the error annotated under the actual failing instruction
    #[track_caller]
    fn annotate_error(&self, instruction_id: InstructionId, mir: &Mir, message: &str) {
        let mir_string = self.display(Some(mir)).to_string();
        let instruction_string = instruction_id.display(self, Some(mir)).to_string();

        // For an instruction string `    foo bar baz`
        // Construct:                `    ^^^^^^^^^^^`
        let trimmed = instruction_string.trim_start();
        let first_non_space = instruction_string.len() - trimmed.len();
        let spaces = " ".repeat(first_non_space);
        let arrows = "^".repeat(trimmed.len() - 1);
        let error_message = format!("{spaces}{arrows} {message}");

        let mut result_string = String::with_capacity(mir_string.len() + instruction_string.len() + 1);

        for (i, s) in mir_string.split(&instruction_string).enumerate() {
            if i != 0 {
                result_string += &instruction_string;
                result_string += &error_message;
                result_string += "\n\n";
            }
            result_string += s;
        }

        panic!("{}", result_string);
    }

    #[rustfmt::skip]
    fn assert_no_unions_or_generics(&self, mir: &Mir) {
        if self.typ.contains_union_or_generic() {
            panic!("{}\nDefinition type contains a union or generic", self.display(Some(mir)));
        }

        for (instruction_id, typ) in self.instruction_result_types.iter() {
            instr_assert!(!typ.contains_union_or_generic(), self, instruction_id, mir, "Result type contains union or generic");
        }

        for (block_id, block) in self.blocks.iter() {
            for parameter in block.parameter_types.iter() {
                if parameter.contains_union_or_generic() {
                    panic!("{}\nParameter to {block_id} contains a union or generic", self.display(Some(mir)));
                }
            }
        }
    }
}

impl Type {
    fn contains_union_or_generic(&self) -> bool {
        match self {
            Type::Primitive(_) => false,
            Type::Union(_) | Type::Generic(_) => true,
            Type::Tuple(fields) => fields.iter().any(Type::contains_union_or_generic),
            Type::Function(function) => {
                function.parameters.iter().any(Type::contains_union_or_generic)
                    || function.return_type.contains_union_or_generic()
            },
        }
    }
}
