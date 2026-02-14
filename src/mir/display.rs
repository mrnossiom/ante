use std::fmt::{Display, Formatter, Result};

use crate::{
    iterator_extensions::mapvec,
    mir::{self, Block, BlockId, DefinitionId, FloatConstant, IntConstant, PrimitiveType, Type, Value},
};

impl Display for mir::Mir {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        for function in self.definitions.values() {
            fmt_function(function, self, f)?;
            writeln!(f, "\n")?;
        }
        Ok(())
    }
}

impl Display for IntConstant {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            IntConstant::U8(x) => write!(f, "{x}_u8"),
            IntConstant::U16(x) => write!(f, "{x}_u16"),
            IntConstant::U32(x) => write!(f, "{x}_u32"),
            IntConstant::U64(x) => write!(f, "{x}_u64"),
            IntConstant::Usz(x) => write!(f, "{x}_usz"),
            IntConstant::I8(x) => write!(f, "{x}_i8"),
            IntConstant::I16(x) => write!(f, "{x}_i16"),
            IntConstant::I32(x) => write!(f, "{x}_i32"),
            IntConstant::I64(x) => write!(f, "{x}_i64"),
            IntConstant::Isz(x) => write!(f, "{x}_isz"),
        }
    }
}

impl Display for FloatConstant {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            FloatConstant::F32(x) => write!(f, "{x}_f32"),
            FloatConstant::F64(x) => write!(f, "{x}_f64"),
        }
    }
}

impl Display for DefinitionId {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "d{}", self.0)
    }
}

impl Display for BlockId {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "b{}", self.0)
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let is_atom = |t: &Type| matches!(t, Type::Primitive(_) | Type::Generic(_) | Type::Union(_));

        match self {
            Type::Primitive(primitive_type) => primitive_type.fmt(f),
            Type::Tuple(items) => {
                let mut type_string =
                    mapvec(items.iter(), |typ| if is_atom(typ) { typ.to_string() } else { format!("({typ})") })
                        .join(", ");

                // Make single-element tuples distinct from other types
                if items.len() == 1 {
                    type_string.push(',');
                }

                if type_string.is_empty() { write!(f, "#empty_tuple") } else { write!(f, "{type_string}") }
            },
            Type::Function(function_type) => {
                write!(f, "fn")?;
                for parameter in &function_type.parameters {
                    write!(f, " ")?;
                    if is_atom(parameter) {
                        write!(f, "{parameter}")?;
                    } else {
                        write!(f, "({parameter})")?;
                    }
                }

                if is_atom(&function_type.return_type) {
                    write!(f, " -> {}", function_type.return_type)
                } else {
                    write!(f, " -> ({})", function_type.return_type)
                }
            },
            Type::Generic(id) => write!(f, "'{}", id.0),
            Type::Union(variants) => {
                write!(f, "{{")?;
                for (i, variant) in variants.iter().enumerate() {
                    if i != 0 {
                        write!(f, " | ")?;
                    }
                    if is_atom(variant) {
                        write!(f, "{variant}")?;
                    } else {
                        write!(f, "({variant})")?;
                    }
                }
                write!(f, "}}")
            },
        }
    }
}

impl Display for PrimitiveType {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            PrimitiveType::Error => write!(f, "error"),
            PrimitiveType::Unit => write!(f, "Unit"),
            PrimitiveType::Bool => write!(f, "Bool"),
            PrimitiveType::Pointer => write!(f, "Pointer"),
            PrimitiveType::Char => write!(f, "Char"),
            PrimitiveType::Int(kind) => kind.fmt(f),
            PrimitiveType::Float(kind) => kind.fmt(f),
        }
    }
}

fn fmt_function(function: &mir::Definition, mir: &mir::Mir, f: &mut Formatter) -> Result {
    write!(f, "fn {} {}: ", function.name, function.id)?;

    if function.generic_count != 0 {
        write!(f, "forall")?;
        for i in 0 .. function.generic_count {
            write!(f, " {}", Type::generic(i))?;
        }
        write!(f, ". ")?;
    }

    write!(f, "{}", function.typ)?;

    for (block_id, block) in function.blocks.iter() {
        writeln!(f)?;
        fmt_block(block_id, mir, function, block, f)?;
    }
    Ok(())
}

fn fmt_block(id: BlockId, mir: &mir::Mir, function: &mir::Definition, block: &Block, f: &mut Formatter) -> Result {
    write!(f, "  b{}(", id.0)?;
    let v = |value: &Value| ValueDisplay { value: *value, mir };

    for (i, typ) in block.parameter_types.iter().enumerate() {
        if i != 0 {
            write!(f, ", ")?;
        }
        write!(f, "{}: {typ}", v(&Value::Parameter(id, i as u32)))?;
    }
    writeln!(f, "):")?;

    for instruction_id in block.instructions.iter().copied() {
        let instruction = &function.instructions[instruction_id];
        fmt_instruction(instruction_id, instruction, mir, function, f)?;
    }

    match block.terminator.as_ref() {
        Some(terminator) => fmt_terminator(terminator, mir, f)?,
        None => write!(f, "  (no terminator)")?,
    }

    Ok(())
}

fn fmt_terminator(terminator: &mir::TerminatorInstruction, mir: &mir::Mir, f: &mut Formatter<'_>) -> Result {
    let v = |value: &Value| ValueDisplay { value: *value, mir };
    write!(f, "    ")?;

    match terminator {
        mir::TerminatorInstruction::Jmp((block_id, argument)) => {
            write!(f, "jmp {block_id}")?;
            if let Some(argument) = argument {
                write!(f, " {}", v(argument))?;
            }
            Ok(())
        },
        mir::TerminatorInstruction::If { condition, then, else_, end } => {
            write!(f, "if {} then {}", v(condition), then.0)?;
            if let Some(argument) = then.1 {
                write!(f, " {}", v(&argument))?;
            }
            write!(f, " else {}", else_.0)?;
            if let Some(argument) = else_.1 {
                write!(f, " {}", v(&argument))?;
            }
            write!(f, " end {end}")
        },
        mir::TerminatorInstruction::Unreachable => write!(f, "unreachable"),
        mir::TerminatorInstruction::Return(value) => write!(f, "return {}", v(value)),
        mir::TerminatorInstruction::Switch { int_value, cases, else_, end } => {
            writeln!(f, "switch {}", v(int_value))?;
            for (i, (case_block, case_arg)) in cases.iter().enumerate() {
                if i != 0 {
                    writeln!(f)?;
                }
                write!(f, "    | {i} -> {case_block}")?;
                if let Some(arg) = case_arg {
                    write!(f, " {}", v(arg))?;
                }
            }
            if let Some((else_block, else_arg)) = else_ {
                write!(f, "\n    | _ -> {else_block}")?;
                if let Some(arg) = else_arg {
                    write!(f, " {}", v(arg))?;
                }
            }
            write!(f, "\n    end {end}")
        },
    }
}

fn fmt_instruction(
    instruction_id: mir::InstructionId, instruction: &mir::Instruction, mir: &mir::Mir, function: &mir::Definition,
    f: &mut Formatter<'_>,
) -> Result {
    let v = |value: &Value| ValueDisplay { value: *value, mir };

    let result_type = &function.instruction_result_types[instruction_id];
    write!(f, "    {}: {result_type} = ", v(&Value::InstructionResult(instruction_id)))?;

    match instruction {
        mir::Instruction::Call { function, arguments } => {
            write!(f, "{}", v(function))?;
            for argument in arguments {
                write!(f, " {}", v(argument))?;
            }
        },
        mir::Instruction::IndexTuple { tuple, index } => write!(f, "{}.{index}", v(tuple))?,
        mir::Instruction::MakeTuple(fields) => write!(f, "({})", comma_separated(fields, mir))?,
        mir::Instruction::MakeString(s) => write!(f, "\"{s}\"")?,
        mir::Instruction::StackAlloc(value) => write!(f, "alloca {}", v(value))?,
        mir::Instruction::Transmute(value) => write!(f, "transmute {}", v(value))?,
        mir::Instruction::Id(value) => write!(f, "id {}", v(value))?,
        mir::Instruction::Instantiate(definition_id, generics) => {
            write!(f, "instantiate {definition_id}")?;
            for generic in generics.iter() {
                write!(f, " {}", generic)?;
            }
        },
    }

    writeln!(f)
}

fn comma_separated(items: &[Value], mir: &mir::Mir) -> String {
    items.iter().map(|v| ValueDisplay { value: *v, mir }.to_string()).collect::<Vec<_>>().join(", ")
}

struct ValueDisplay<'local> {
    value: Value,
    mir: &'local mir::Mir,
}

impl<'local> Display for ValueDisplay<'local> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match &self.value {
            Value::Error => write!(f, "#error"),
            Value::Unit => write!(f, "()"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Char(c) => write!(f, "{c}"),
            Value::Integer(int) => write!(f, "{int}"),
            Value::Float(float) => write!(f, "{float}"),
            Value::InstructionResult(instruction_id) => write!(f, "v{}", instruction_id.0),
            Value::Parameter(block_id, i) => write!(f, "b{}_{}", block_id.0, i),
            Value::Definition(id) => {
                if let Some(name) = self.mir.names.get(id) {
                    write!(f, "{name}_")?;
                }
                write!(f, "{id}")
            },
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Value::Error => write!(f, "#error"),
            Value::Unit => write!(f, "()"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Char(c) => write!(f, "{c}"),
            Value::Integer(int) => write!(f, "{int}"),
            Value::Float(float) => write!(f, "{float}"),
            Value::InstructionResult(instruction_id) => write!(f, "v{}", instruction_id.0),
            Value::Parameter(block_id, i) => write!(f, "b{}_{}", block_id.0, i),
            Value::Definition(id) => write!(f, "{id}"),
        }
    }
}
