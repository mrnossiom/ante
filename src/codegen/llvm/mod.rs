use std::sync::Arc;

use inkwell::{
    AddressSpace, FloatPredicate, IntPredicate,
    basic_block::BasicBlock,
    builder::Builder,
    memory_buffer::MemoryBuffer,
    module::Module,
    targets::{CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine},
    types::{BasicType, BasicTypeEnum, IntType},
    values::{AggregateValue, AnyValue, BasicValue, BasicValueEnum, FunctionValue, GlobalValue},
};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::{
    incremental::Db,
    iterator_extensions::mapvec,
    lexer::token::{FloatKind, IntegerKind},
    mir::{self, BlockId, DefinitionId, FloatConstant, PrimitiveType, TerminatorInstruction},
    vecmap::VecMap,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodegenLlvmResult {
    pub module_bitcode: Arc<Vec<u8>>,
}

pub fn initialize_native_target() {
    let config = InitializationConfig::default();
    Target::initialize_native(&config).unwrap();
}

pub fn codegen_llvm(compiler: &Db) -> Option<CodegenLlvmResult> {
    // Monomorphize everything - ideally we could only monomorphize some items so that the
    // `CodegenLlvmResult` later can be split up and combined but for now it is whole program only.
    let mir = mir::monomorphization::monomorphize(compiler);
    let name = &mir.definitions.iter().next().map_or("_", |(_, function)| &function.name);

    initialize_native_target();
    let llvm = inkwell::context::Context::create();
    let mut module = ModuleContext::new(&llvm, &mir, name);

    for (id, function) in &mir.definitions {
        module.codegen_function(function, *id);
    }

    for (id, typ) in &mir.externals {
        module.codegen_extern(*id, typ);
    }

    if let Err(error) = module.module.verify() {
        module.module.print_to_stderr();
        eprintln!("llvm module failed to verify: {error}");
    }

    // TODO: This is inefficient
    let bitcode = module.module.write_bitcode_to_memory();
    let bitcode = bitcode.as_slice().to_vec();
    let module_bitcode = Arc::new(bitcode);

    // Compiling each function separately and linking them together later is probably slower than
    // doing them all together to begin with but oh well. It is easier to start more incremental
    // and be less incremental later than the reverse.
    Some(CodegenLlvmResult { module_bitcode })
}

/// Link the given list of llvm bitcode modules into an executable.
pub fn link(modules: Vec<Arc<Vec<u8>>>, binary_name: &str) {
    let llvm = inkwell::context::Context::create();
    let module = llvm.create_module(binary_name);

    for bitcode in modules {
        let buffer = MemoryBuffer::create_from_memory_range(&bitcode, "buffer");
        let new_module =
            Module::parse_bitcode_from_buffer(&buffer, &llvm).expect("Failed to parse llvm module bitcode");
        module.link_in_module(new_module).expect("Failed to link in llvm module");
    }

    // generate the bitcode to a .bc file
    let path = std::path::Path::new(binary_name).with_extension("o");
    let target_machine = native_target_machine();

    target_machine.write_to_file(&module, FileType::Object, &path).unwrap();

    // call gcc to compile the bitcode to a binary
    super::link_with_gcc(path.to_string_lossy().as_ref(), binary_name);
}

fn native_target_machine() -> TargetMachine {
    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple).unwrap();
    target
        .create_target_machine(&triple, "", "", inkwell::OptimizationLevel::None, RelocMode::PIC, CodeModel::Default)
        .unwrap()
}

struct ModuleContext<'ctx> {
    llvm: &'ctx inkwell::context::Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,

    mir: &'ctx mir::Mir,

    blocks: VecMap<BlockId, BasicBlock<'ctx>>,

    current_function: Option<DefinitionId>,
    current_function_value: Option<FunctionValue<'ctx>>,

    global_values: FxHashMap<mir::Value, GlobalValue<'ctx>>,
    values: FxHashMap<mir::Value, BasicValueEnum<'ctx>>,

    /// Block arguments are added here to later insert them as PHI values.
    ///
    /// Maps merge_block to a vec of each incoming block along with the arguments it branches with.
    incoming: FxHashMap<BlockId, Vec<(BasicBlock<'ctx>, BasicValueEnum<'ctx>)>>,
}

impl<'ctx> ModuleContext<'ctx> {
    fn new(llvm: &'ctx inkwell::context::Context, mir: &'ctx mir::Mir, name: &str) -> Self {
        let module = llvm.create_module(name);
        let target = TargetMachine::get_default_triple();
        module.set_triple(&target);
        Self {
            llvm,
            module,
            mir,
            current_function: None,
            current_function_value: None,
            global_values: Default::default(),
            values: Default::default(),
            builder: llvm.create_builder(),
            blocks: Default::default(),
            incoming: Default::default(),
        }
    }

    fn codegen_global(&mut self, global: &mir::Definition, id: mir::DefinitionId) {
        let typ = self.convert_type(&global.typ);
        let name = self.get_name(id);
        let global_var = self.module.add_global(typ, None, name);

        // Save the current values map (may be non-empty if called from within function codegen)
        let saved_values = std::mem::take(&mut self.values);

        // Evaluate each instruction in the entry block as a constant
        for instr_id in global.entry_block().instructions.iter().copied() {
            let value = self.codegen_constant_instruction(global, instr_id);
            self.values.insert(mir::Value::InstructionResult(instr_id), value);
        }

        // Get the Result value and use it as the initializer
        let TerminatorInstruction::Result(result_value) = global.entry_block().terminator.as_ref().unwrap() else {
            panic!("Global definition missing Result terminator");
        };
        let initializer = self.constant_value(*result_value);
        global_var.set_initializer(&initializer);

        // Restore the saved values map
        self.values = saved_values;

        self.global_values.insert(mir::Value::Definition(id), global_var);
    }

    fn constant_value(&mut self, value: mir::Value) -> BasicValueEnum<'ctx> {
        match value {
            mir::Value::Unit => self.llvm.const_struct(&[], false).into(),
            mir::Value::Bool(b) => self.llvm.bool_type().const_int(b as u64, false).into(),
            mir::Value::Char(c) => self.llvm.i32_type().const_int(c as u64, false).into(),
            mir::Value::Integer(constant) => {
                let kind = constant.kind();
                let typ = self.convert_integer_kind(kind);
                typ.const_int(constant.as_u64(), kind.is_signed()).into()
            },
            mir::Value::Float(FloatConstant::F32(v)) => self.llvm.f32_type().const_float(v.0).into(),
            mir::Value::Float(FloatConstant::F64(v)) => self.llvm.f64_type().const_float(v.0).into(),
            mir::Value::InstructionResult(_) | mir::Value::Parameter(..) => {
                *self.values.get(&value).expect("constant value not cached")
            },
            mir::Value::Definition(id) => {
                if let Some(gv) = self.global_values.get(&mir::Value::Definition(id)) {
                    return gv.as_pointer_value().into();
                }
                let def = self.mir.definitions.get(&id).expect("constant_value: definition not found");
                let fn_type = self
                    .convert_function_type(&def.typ)
                    .expect("constant_value: definition in global initializer is not a function");
                let name = self.get_name(id);
                let fn_val = self.module.add_function(name, fn_type, None);
                let gv = fn_val.as_global_value();
                self.global_values.insert(mir::Value::Definition(id), gv);
                gv.as_pointer_value().into()
            },
            mir::Value::Error => unreachable!("Error value in global initializer"),
        }
    }

    fn codegen_constant_instruction(
        &mut self, global: &mir::Definition, id: mir::InstructionId,
    ) -> BasicValueEnum<'ctx> {
        match &global.instructions[id] {
            mir::Instruction::MakeTuple(fields) => {
                let fields: Vec<mir::Value> = fields.clone();
                let field_values: Vec<BasicValueEnum<'ctx>> = fields.iter().map(|f| self.constant_value(*f)).collect();
                self.llvm.const_struct(&field_values, false).into()
            },
            mir::Instruction::Id(value) => {
                let value = *value;
                self.constant_value(value)
            },
            _ => panic!("Unsupported instruction in global initializer"),
        }
    }

    fn codegen_function(&mut self, function: &mir::Definition, id: mir::DefinitionId) {
        if function.is_global() {
            self.codegen_global(function, id);
            return;
        }

        let function_value = match self.global_values.get(&mir::Value::Definition(id)) {
            Some(existing) => existing.as_any_value_enum().into_function_value(),
            None => {
                let function_type = self.convert_function_type(&function.typ).unwrap();
                let function_value = self.module.add_function(&function.name, function_type, None);
                self.global_values.insert(mir::Value::Definition(id), function_value.as_global_value());
                function_value
            },
        };

        self.current_function = Some(id);
        self.current_function_value = Some(function_value);
        self.create_blocks(function, function_value);

        for i in 0..function.blocks[BlockId::ENTRY_BLOCK].parameter_types.len() as u32 {
            let value = mir::Value::Parameter(BlockId::ENTRY_BLOCK, i);
            let llvm_value = function_value.get_nth_param(i).unwrap();
            self.values.insert(value, llvm_value);
        }

        for block in function.topological_sort() {
            self.codegen_block(block, function);
        }

        self.values.clear();
        self.blocks.clear();
    }

    /// Create an empty block for each block in the given function
    fn create_blocks(&mut self, function: &mir::Definition, function_value: FunctionValue<'ctx>) {
        for (block_id, _) in function.blocks.iter() {
            let block = self.llvm.append_basic_block(function_value, "");
            self.blocks.push_existing(block_id, block);
        }
    }

    fn codegen_block(&mut self, block_id: BlockId, function: &mir::Definition) {
        let llvm_block = self.blocks[block_id];
        self.builder.position_at_end(llvm_block);
        let block = &function.blocks[block_id];

        // Translate the block parameters into phi instructions
        if block_id != BlockId::ENTRY_BLOCK {
            for (parameter, parameter_type) in block.parameters(block_id) {
                let parameter_type = self.convert_type(&parameter_type);
                let phi = self.builder.build_phi(parameter_type, "").unwrap();

                let incoming = self
                    .incoming
                    .remove(&block_id)
                    .unwrap_or_else(|| panic!("llvm codegen: No incoming for block {block_id}"));

                for (block, block_args) in incoming {
                    phi.add_incoming(&[(&block_args, block)]);
                }
                self.values.insert(parameter, phi.as_basic_value());
            }
        }

        for instruction_id in block.instructions.iter().copied() {
            self.codegen_instruction(function, instruction_id);
        }

        let terminator = block.terminator.as_ref().expect("Incomplete MIR: missing block terminator");
        self.codegen_terminator(terminator);
    }

    fn convert_type(&self, typ: &mir::Type) -> BasicTypeEnum<'ctx> {
        match typ {
            mir::Type::Primitive(primitive_type) => self.convert_primitive_type(*primitive_type),
            mir::Type::Tuple(fields) => {
                let fields = mapvec(fields.iter(), |typ| self.convert_type(typ));
                let struct_type = self.llvm.struct_type(&fields, false);
                BasicTypeEnum::StructType(struct_type)
            },
            mir::Type::Function(_) => self.llvm.ptr_type(AddressSpace::default()).into(),
            mir::Type::Union(_) => self.llvm.ptr_type(AddressSpace::default()).into(),
            mir::Type::Generic(_) => self.llvm.ptr_type(AddressSpace::default()).into(),
        }
    }

    fn convert_primitive_type(&self, primitive_type: PrimitiveType) -> BasicTypeEnum<'ctx> {
        match primitive_type {
            PrimitiveType::Error => self.llvm.struct_type(&[], false).into(), //unreachable!("Cannot codegen llvm with errors"),
            PrimitiveType::Unit => self.llvm.struct_type(&[], false).into(),
            PrimitiveType::Bool => self.llvm.bool_type().into(),
            PrimitiveType::Pointer => self.llvm.ptr_type(AddressSpace::default()).into(),
            PrimitiveType::Char => self.llvm.i32_type().into(),
            PrimitiveType::Int(kind) => self.convert_integer_kind(kind).into(),
            PrimitiveType::Float(FloatKind::F32) => self.llvm.f32_type().into(),
            PrimitiveType::Float(FloatKind::F64) => self.llvm.f64_type().into(),
        }
    }

    fn convert_integer_kind(&self, kind: IntegerKind) -> IntType<'ctx> {
        match kind {
            IntegerKind::I8 | IntegerKind::U8 => self.llvm.i8_type(),
            IntegerKind::I16 | IntegerKind::U16 => self.llvm.i16_type(),
            IntegerKind::I32 | IntegerKind::U32 => self.llvm.i32_type(),
            IntegerKind::I64 | IntegerKind::U64 => self.llvm.i64_type(),
            IntegerKind::Isz | IntegerKind::Usz => {
                let machine = native_target_machine();
                let target_data = machine.get_target_data();
                self.llvm.ptr_sized_int_type(&target_data, None)
            },
        }
    }

    /// Convert a type into a function type, returns None if the given type is not a function.
    /// When passed to [Self::convert_type], function types are translated to pointers by default,
    /// necessitating this function when an actual function type is required.
    fn convert_function_type(&self, typ: &mir::Type) -> Option<inkwell::types::FunctionType<'ctx>> {
        let mir::Type::Function(function_type) = typ else {
            return None;
        };

        let return_type = self.convert_type(&function_type.return_type);
        let parameters = mapvec(&function_type.parameters, |parameter| self.convert_type(parameter).into());
        Some(return_type.fn_type(&parameters, false))
    }

    /// Return the given [mir::Definition] if it is within `self.mir`. Return `None` otherwise.
    fn try_get_function(&self, id: DefinitionId) -> Option<&'ctx mir::Definition> {
        self.mir.definitions.get(&id)
    }

    /// Returns the name of the given [DefinitionId].
    /// As long as the [DefinitionId] is referenced in `self.mir`, this should never panic.
    fn get_name(&self, id: DefinitionId) -> &'ctx str {
        self.mir.get_name(id).unwrap().as_ref()
    }

    fn lookup_value(&mut self, value: mir::Value) -> BasicValueEnum<'ctx> {
        match value {
            mir::Value::Error => unreachable!("Error value encountered during llvm codegen"),
            mir::Value::Unit => self.llvm.const_struct(&[], false).into(),
            mir::Value::Bool(value) => self.llvm.bool_type().const_int(value as u64, false).into(),
            mir::Value::Char(value) => self.llvm.i32_type().const_int(value as u64, false).into(),
            mir::Value::Integer(constant) => {
                let kind = constant.kind();
                let typ = self.convert_integer_kind(kind);
                typ.const_int(constant.as_u64(), kind.is_signed()).into()
            },
            mir::Value::Float(FloatConstant::F32(value)) => self.llvm.f32_type().const_float(value.0).into(),
            mir::Value::Float(FloatConstant::F64(value)) => self.llvm.f64_type().const_float(value.0).into(),
            mir::Value::InstructionResult(_) | mir::Value::Parameter(..) => {
                *self.values.get(&value).unwrap_or_else(|| panic!("llvm codegen: mir value is not cached: {value}"))
            },
            mir::Value::Definition(function_id) => {
                if let Some(gv) = self.global_values.get(&value) {
                    return gv.as_pointer_value().into();
                }

                let function = self.try_get_function(self.current_function.unwrap()).unwrap();
                let typ = self.mir.type_of_value(&value, function);

                if let Some(fn_type) = self.convert_function_type(&typ) {
                    let name = self.get_name(function_id);
                    let function_value = self.module.add_function(name, fn_type, None).as_global_value();
                    self.global_values.insert(value, function_value);
                    function_value.as_pointer_value().into()
                } else {
                    // It's a global variable — codegen it now if not yet done
                    let def =
                        self.mir.definitions.get(&function_id).expect("lookup_value: definition not found").clone();
                    self.codegen_global(&def, function_id);
                    let global_var = self.global_values[&value];
                    let llvm_type = self.convert_type(&typ);
                    self.builder.build_load(llvm_type, global_var.as_pointer_value(), "").unwrap()
                }
            },
        }
    }

    fn codegen_instruction(&mut self, function: &mir::Definition, id: mir::InstructionId) {
        let result = match &function.instructions[id] {
            mir::Instruction::Call { function: function_value, arguments } => {
                let typ = self.convert_function_type(&self.mir.type_of_value(function_value, function)).unwrap();
                let function = self.lookup_value(*function_value).into_pointer_value();
                let arguments = mapvec(arguments, |arg| self.lookup_value(*arg).into());
                self.builder
                    .build_indirect_call(typ, function, &arguments, "")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_basic()
            },
            mir::Instruction::IndexTuple { tuple, index } => {
                let tuple = self.lookup_value(*tuple).into_struct_value();
                self.builder.build_extract_value(tuple, *index, "").unwrap()
            },
            mir::Instruction::MakeString(string) => {
                let string_data = self.llvm.const_string(string.as_bytes(), false);
                // Need to create a unique name across modules. Llvm will auto-rename within
                // the same module so duplicate names within one function won't matter.
                let name = format!("{}_str", self.current_function.unwrap());
                let global = self.module.add_global(string_data.get_type(), None, &name);
                global.set_initializer(&string_data);
                let c_string = global.as_pointer_value().into();

                let length = self.llvm.i32_type().const_int(string.len() as u64, false).into();
                self.llvm.const_struct(&[c_string, length], false).into()
            },
            mir::Instruction::MakeTuple(fields) => {
                let fields = mapvec(fields, |field| self.lookup_value(*field));
                let const_fields =
                    mapvec(
                        &fields,
                        |field| if field.is_const() { *field } else { Self::undef_value(field.get_type()) },
                    );
                let mut tuple = self.llvm.const_struct(&const_fields, false).as_aggregate_value_enum();

                for (i, field) in fields.into_iter().enumerate() {
                    if !field.is_const() {
                        tuple = self.builder.build_insert_value(tuple, field, i as u32, "").unwrap();
                    }
                }
                tuple.as_basic_value_enum()
            },
            mir::Instruction::StackAlloc(value) => {
                let value = self.lookup_value(*value);
                let alloca = self.builder.build_alloca(value.get_type(), "").unwrap();
                self.builder.build_store(alloca, value).unwrap();
                alloca.into()
            },
            mir::Instruction::Transmute(value) => {
                // Transmute the value by storing it in an alloca and loading it as a different type
                let result_type = self.convert_type(function.instruction_result_type(id));
                let value = self.lookup_value(*value);
                let alloca = self.builder.build_alloca(value.get_type(), "").unwrap();
                self.builder.build_store(alloca, value).unwrap();
                self.builder.build_load(result_type, alloca, "").unwrap()
            },
            mir::Instruction::Id(value) => self.lookup_value(*value),
            mir::Instruction::Instantiate(..) => {
                unreachable!("Instruction::Instantiate remaining in the code during llvm codegen")
            },
            mir::Instruction::AddInt(a, b) => {
                let a = self.lookup_value(*a).into_int_value();
                let b = self.lookup_value(*b).into_int_value();
                self.builder.build_int_add(a, b, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::AddFloat(a, b) => {
                let a = self.lookup_value(*a).into_float_value();
                let b = self.lookup_value(*b).into_float_value();
                self.builder.build_float_add(a, b, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::SubInt(a, b) => {
                let a = self.lookup_value(*a).into_int_value();
                let b = self.lookup_value(*b).into_int_value();
                self.builder.build_int_sub(a, b, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::SubFloat(a, b) => {
                let a = self.lookup_value(*a).into_float_value();
                let b = self.lookup_value(*b).into_float_value();
                self.builder.build_float_sub(a, b, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::MulInt(a, b) => {
                let a = self.lookup_value(*a).into_int_value();
                let b = self.lookup_value(*b).into_int_value();
                self.builder.build_int_mul(a, b, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::MulFloat(a, b) => {
                let a = self.lookup_value(*a).into_float_value();
                let b = self.lookup_value(*b).into_float_value();
                self.builder.build_float_mul(a, b, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::DivSigned(a, b) => {
                let a = self.lookup_value(*a).into_int_value();
                let b = self.lookup_value(*b).into_int_value();
                self.builder.build_int_signed_div(a, b, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::DivUnsigned(a, b) => {
                let a = self.lookup_value(*a).into_int_value();
                let b = self.lookup_value(*b).into_int_value();
                self.builder.build_int_unsigned_div(a, b, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::DivFloat(a, b) => {
                let a = self.lookup_value(*a).into_float_value();
                let b = self.lookup_value(*b).into_float_value();
                self.builder.build_float_div(a, b, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::ModSigned(a, b) => {
                let a = self.lookup_value(*a).into_int_value();
                let b = self.lookup_value(*b).into_int_value();
                self.builder.build_int_signed_div(a, b, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::ModUnsigned(a, b) => {
                let a = self.lookup_value(*a).into_int_value();
                let b = self.lookup_value(*b).into_int_value();
                self.builder.build_int_unsigned_div(a, b, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::ModFloat(a, b) => {
                let a = self.lookup_value(*a).into_float_value();
                let b = self.lookup_value(*b).into_float_value();
                self.builder.build_float_rem(a, b, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::LessSigned(a, b) => {
                let a = self.lookup_value(*a).into_int_value();
                let b = self.lookup_value(*b).into_int_value();
                self.builder.build_int_compare(IntPredicate::SLT, a, b, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::LessUnsigned(a, b) => {
                let a = self.lookup_value(*a).into_int_value();
                let b = self.lookup_value(*b).into_int_value();
                self.builder.build_int_compare(IntPredicate::ULT, a, b, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::LessFloat(a, b) => {
                let a = self.lookup_value(*a).into_float_value();
                let b = self.lookup_value(*b).into_float_value();
                self.builder.build_float_compare(FloatPredicate::OLT, a, b, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::EqInt(a, b) => {
                let a = self.lookup_value(*a).into_int_value();
                let b = self.lookup_value(*b).into_int_value();
                self.builder.build_int_compare(IntPredicate::EQ, a, b, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::EqFloat(a, b) => {
                let a = self.lookup_value(*a).into_float_value();
                let b = self.lookup_value(*b).into_float_value();
                self.builder.build_float_compare(FloatPredicate::OEQ, a, b, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::BitwiseAnd(a, b) => {
                let a = self.lookup_value(*a).into_int_value();
                let b = self.lookup_value(*b).into_int_value();
                self.builder.build_and(a, b, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::BitwiseOr(a, b) => {
                let a = self.lookup_value(*a).into_int_value();
                let b = self.lookup_value(*b).into_int_value();
                self.builder.build_or(a, b, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::BitwiseXor(a, b) => {
                let a = self.lookup_value(*a).into_int_value();
                let b = self.lookup_value(*b).into_int_value();
                self.builder.build_xor(a, b, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::BitwiseNot(value) => {
                let value = self.lookup_value(*value).into_int_value();
                self.builder.build_not(value, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::SignExtend(value) => {
                let value = self.lookup_value(*value).into_int_value();
                let int_type = self.convert_type(function.instruction_result_type(id)).into_int_type();
                self.builder.build_int_s_extend(value, int_type, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::ZeroExtend(value) => {
                let value = self.lookup_value(*value).into_int_value();
                let int_type = self.convert_type(function.instruction_result_type(id)).into_int_type();
                self.builder.build_int_z_extend(value, int_type, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::SignedToFloat(value) => {
                let value = self.lookup_value(*value).into_int_value();
                let float_type = self.convert_type(function.instruction_result_type(id)).into_float_type();
                self.builder.build_signed_int_to_float(value, float_type, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::UnsignedToFloat(value) => {
                let value = self.lookup_value(*value).into_int_value();
                let float_type = self.convert_type(function.instruction_result_type(id)).into_float_type();
                self.builder.build_unsigned_int_to_float(value, float_type, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::FloatToSigned(value) => {
                let value = self.lookup_value(*value).into_float_value();
                let int_type = self.convert_type(function.instruction_result_type(id)).into_int_type();
                self.builder.build_float_to_signed_int(value, int_type, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::FloatToUnsigned(value) => {
                let value = self.lookup_value(*value).into_float_value();
                let int_type = self.convert_type(function.instruction_result_type(id)).into_int_type();
                self.builder.build_float_to_unsigned_int(value, int_type, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::FloatPromote(value) => {
                let value = self.lookup_value(*value).into_float_value();
                let float_type = self.convert_type(function.instruction_result_type(id)).into_float_type();
                self.builder.build_float_cast(value, float_type, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::FloatDemote(value) => {
                let value = self.lookup_value(*value).into_float_value();
                let float_type = self.convert_type(function.instruction_result_type(id)).into_float_type();
                self.builder.build_float_cast(value, float_type, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::Truncate(value) => {
                let value = self.lookup_value(*value).into_int_value();
                let int_type = self.convert_type(function.instruction_result_type(id)).into_int_type();
                self.builder.build_int_truncate(value, int_type, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::Deref(value) => {
                let value = self.lookup_value(*value).into_pointer_value();
                let result_type = self.convert_type(function.instruction_result_type(id));
                self.builder.build_load(result_type, value, "").unwrap().as_basic_value_enum()
            },
            mir::Instruction::SizeOf(_) => todo!("SizeOf should be removed by monomorphization"),
        };
        self.values.insert(mir::Value::InstructionResult(id), result);
    }

    fn undef_value(typ: BasicTypeEnum<'ctx>) -> BasicValueEnum<'ctx> {
        match typ {
            BasicTypeEnum::ArrayType(array) => array.get_undef().into(),
            BasicTypeEnum::FloatType(float) => float.get_undef().into(),
            BasicTypeEnum::IntType(int) => int.get_undef().into(),
            BasicTypeEnum::PointerType(pointer) => pointer.get_undef().into(),
            BasicTypeEnum::StructType(tuple) => tuple.get_undef().into(),
            BasicTypeEnum::VectorType(vector) => vector.get_undef().into(),
            BasicTypeEnum::ScalableVectorType(vector) => vector.get_undef().into(),
        }
    }

    fn remember_incoming(&mut self, target: BlockId, argument: &Option<mir::Value>) {
        if let Some(argument) = argument {
            let current_block = self.builder.get_insert_block().unwrap();
            let argument = self.lookup_value(*argument);
            self.incoming.entry(target).or_default().push((current_block, argument));
        }
    }

    fn codegen_terminator(&mut self, terminator: &TerminatorInstruction) {
        match terminator {
            TerminatorInstruction::Jmp((target_id, argument)) => {
                let target = self.blocks[*target_id];
                self.builder.build_unconditional_branch(target).unwrap();
                self.remember_incoming(*target_id, argument);
            },
            TerminatorInstruction::If { condition, then, else_, end: _ } => {
                let condition = self.lookup_value(*condition).into_int_value();

                let then_target = self.blocks[then.0];
                let else_target = self.blocks[else_.0];
                self.builder.build_conditional_branch(condition, then_target, else_target).unwrap();

                self.remember_incoming(then.0, &then.1);
                self.remember_incoming(else_.0, &else_.1);
            },
            TerminatorInstruction::Switch { int_value, cases, else_, end: _ } => {
                let int_value = self.lookup_value(*int_value).into_int_value();

                let cases = mapvec(cases.iter().enumerate(), |(i, (case_block, case_args))| {
                    self.remember_incoming(*case_block, case_args);
                    let case_block = self.blocks[*case_block];
                    let int_value = int_value.get_type().const_int(i as u64, false);
                    (int_value, case_block)
                });

                let else_block = if let Some((else_block, args)) = else_ {
                    self.remember_incoming(*else_block, args);
                    self.blocks[*else_block]
                } else {
                    // No else block but switch in llvm requires one.
                    // Create an empty block with an `unreachable` terminator.
                    let block = self.llvm.append_basic_block(self.current_function_value.unwrap(), "");
                    let current_block = self.builder.get_insert_block().unwrap();
                    self.builder.position_at_end(block);
                    self.builder.build_unreachable().unwrap();
                    self.builder.position_at_end(current_block);
                    block
                };

                self.builder.build_switch(int_value, else_block, &cases).unwrap();
            },
            TerminatorInstruction::Unreachable => {
                self.builder.build_unreachable().unwrap();
            },
            TerminatorInstruction::Return(value) => {
                let value = self.lookup_value(*value);
                self.builder.build_return(Some(&value)).unwrap();
            },
            TerminatorInstruction::Result(_) => {
                unreachable!("Result terminator encountered during function codegen")
            },
        }
    }

    fn codegen_extern(&mut self, id: DefinitionId, external: &mir::Extern) {
        if !self.values.contains_key(&mir::Value::Definition(id)) {
            let function_type = self.convert_function_type(&external.typ).unwrap();
            let name = &external.name;
            let function_value = self.module.add_function(name, function_type, None);
            self.values.insert(mir::Value::Definition(id), function_value.as_global_value().as_pointer_value().into());
        }
    }
}
