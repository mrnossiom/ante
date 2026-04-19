//! Lower remaining [crate::mir::Instruction::Handle] and
//! [crate::mir::Instruction::Perform] instructions into aminicoro primitives.
//!
//! For each `Handle { body, cases }` we emit:
//! 1. A heap-allocated slot holding `body` (the closure tuple).
//! 2. `coro = mco_coro_init(body_entry_wrapper, &body_slot)`.
//! 3. A call into a per-Handle-site `drive` function that:
//!   - `mco_coro_resume coro`
//!   - if not suspended: pop R, return it (and the caller then frees).
//!   - if suspended: pop an `op_tag: u32`, switch on every effect op in the program:
//!     - handled ops: pop op's args, invoke the handler with `(args.., resume_closure)`, jmp to final.
//!     - unhandled ops (handled somewhere up the stack): pop op's args, re-push them onto
//!       `mco_coro_running ()` (the outer coro, since this inner is currently suspended), suspend
//!       that outer, pop the resumed value back, push it onto this inner, recurse into `drive` to
//!       keep this inner going.
//! 4. `mco_coro_free coro` in the caller.
//!
//! Each `resume` is a free function that takes a value, pushes it onto `coro`, and recurses back
//! into `drive`. Any further effects raised by the resumed body are handled by the same `drive`.
//!
//! Each `Perform(op, args)` becomes:
//!   running = mco_coro_running ()
//!   push running args..
//!   push running op_tag
//!   suspend running
//!   pop running (ref result)  // the resumed value
//!
//! Each effect's tag is keyed by its [DefinitionId], meaning we don't handle multiple effects of
//! the same kind but with different type arguments (like `State U32` and `State (Vec t)`)
use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::{
    iterator_extensions::mapvec,
    lexer::token::IntegerKind,
    mir::{
        Block, BlockId, Definition, DefinitionId, FunctionType, HandlerCase, Instruction, InstructionId, IntConstant,
        Mir, PrimitiveType, TerminatorInstruction, Type, Value, next_definition_id,
    },
};

struct AminicoroFn {
    name: &'static str,
    typ: Type,
}

struct AminicoroFns {
    init: AminicoroFn,
    free: AminicoroFn,
    is_suspended: AminicoroFn,
    push: AminicoroFn,
    pop: AminicoroFn,
    suspend: AminicoroFn,
    resume: AminicoroFn,
    running: AminicoroFn,
    get_user_data: AminicoroFn,
}

fn ptr_fn(parameters: Vec<Type>, return_type: Type) -> Type {
    Type::Function(Arc::new(FunctionType { parameters, environment: Type::NO_CLOSURE_ENV, return_type }))
}

fn aminicoro_fns() -> AminicoroFns {
    let ptr = || Type::POINTER;
    let u8_t = || Type::int(IntegerKind::U8);
    let usize_t = || Type::int(IntegerKind::Usz);

    AminicoroFns {
        init: AminicoroFn { name: "mco_coro_init", typ: ptr_fn(vec![ptr(), ptr()], ptr()) },
        free: AminicoroFn { name: "mco_coro_free", typ: ptr_fn(vec![ptr()], u8_t()) },
        is_suspended: AminicoroFn { name: "mco_coro_is_suspended", typ: ptr_fn(vec![ptr()], Type::BOOL) },
        push: AminicoroFn { name: "mco_coro_push", typ: ptr_fn(vec![ptr(), ptr(), usize_t()], u8_t()) },
        pop: AminicoroFn { name: "mco_coro_pop", typ: ptr_fn(vec![ptr(), ptr(), usize_t()], u8_t()) },
        suspend: AminicoroFn { name: "mco_coro_suspend", typ: ptr_fn(vec![ptr()], u8_t()) },
        resume: AminicoroFn { name: "mco_coro_resume", typ: ptr_fn(vec![ptr()], u8_t()) },
        running: AminicoroFn { name: "mco_coro_running", typ: ptr_fn(vec![], ptr()) },
        get_user_data: AminicoroFn { name: "mco_coro_get_user_data", typ: ptr_fn(vec![ptr()], ptr()) },
    }
}

impl Mir {
    pub(crate) fn lower_effects(mut self, ptr_size: u32) -> Self {
        if !contains_effects(&self) {
            return self;
        }
        let fns = aminicoro_fns();
        let op_table = build_op_table(&self);
        let context = Context { mco: &fns, op_table: &op_table, ptr_size };

        let definition_ids: Vec<DefinitionId> = self.definitions.keys().copied().collect();

        // First pass: rewrite every `Perform` into push/suspend/pop. Does not
        // require knowing which Handle catches the Perform because we rely on
        // `mco_running()` at the call site to recover the correct coroutine.
        for id in &definition_ids {
            rewrite_sites_in_definition(&mut self, *id, context, collect_perform_sites, rewrite_single_perform);
        }

        // Second pass: rewrite every `Handle` by generating trampolines and
        // splicing init/drive/free in place of the Handle instruction.
        for id in &definition_ids {
            rewrite_sites_in_definition(&mut self, *id, context, collect_handle_sites, rewrite_single_handle);
        }

        self
    }
}

/// Walk every instruction in `definition`, calling `extract` on each. Sites
/// are grouped per block and reversed within each block so later splices
/// don't invalidate earlier indices.
fn collect_sites<S>(
    definition: &Definition, mut extract: impl FnMut(BlockId, usize, InstructionId) -> Option<S>,
) -> Vec<S> {
    let mut sites = Vec::new();
    for (block_id, block) in definition.blocks.iter() {
        let start = sites.len();
        for (index, instruction_id) in block.instructions.iter().enumerate() {
            if let Some(site) = extract(block_id, index, *instruction_id) {
                sites.push(site);
            }
        }
        sites[start..].reverse();
    }
    sites
}

fn rewrite_sites_in_definition<S>(
    mir: &mut Mir, definition_id: DefinitionId, context: Context, collect: impl FnOnce(&Definition) -> Vec<S>,
    mut rewrite: impl FnMut(&mut Mir, DefinitionId, S, Context),
) {
    let Some(definition) = mir.definitions.get(&definition_id) else { return };
    let sites = collect(definition);
    for site in sites {
        rewrite(mir, definition_id, site, context);
    }
}

fn contains_effects(mir: &Mir) -> bool {
    mir.definitions.values().any(|definition| {
        definition
            .instructions
            .values()
            .any(|instruction| matches!(instruction, Instruction::Perform { .. } | Instruction::Handle { .. }))
    })
}

#[derive(Clone, Copy)]
struct Context<'local> {
    mco: &'local AminicoroFns,
    op_table: &'local OpTable,
    ptr_size: u32,
}

struct Effect {
    tag: u32,
    // In practice all effect operators should have a signature
    signature: Option<EffectSignature>,
}

struct EffectSignature {
    arg_types: Vec<Type>,
    return_type: Type,
}

type OpTable = FxHashMap<DefinitionId, Effect>;

fn build_op_table(mir: &Mir) -> OpTable {
    let mut table: OpTable = FxHashMap::default();
    let mut next_tag = 0;

    for definition in mir.definitions.values() {
        for instruction in definition.instructions.values() {
            match instruction {
                Instruction::Perform { effect_op, .. } => {
                    assign_tag(&mut table, &mut next_tag, *effect_op);
                },
                Instruction::Handle { cases, .. } => {
                    for case in cases {
                        assign_tag(&mut table, &mut next_tag, case.effect_op);
                        let handler_type = definition.type_of_value(&case.handler, &mir.externals, &mir.definitions);
                        let Type::Function(ft) = &handler_type else { continue };
                        let Some((_resume, arg_types)) = ft.parameters.split_last() else { continue };
                        let return_type = resume_arg_type_from_handler(&handler_type).unwrap_or(Type::UNIT);
                        if let Some(info) = table.get_mut(&case.effect_op) {
                            info.signature = Some(EffectSignature { arg_types: arg_types.to_vec(), return_type });
                        }
                    }
                },
                _ => {},
            }
        }
    }

    table
}

fn assign_tag(table: &mut OpTable, next_tag: &mut u64, op: DefinitionId) {
    table.entry(op).or_insert_with(|| {
        let tag = u32::try_from(*next_tag)
            .unwrap_or_else(|_| panic!("effect_lowering: more than u32::MAX distinct effect ops in this program"));
        *next_tag += 1;
        Effect { tag, signature: None }
    });
}

fn op_tag(table: &OpTable, op: DefinitionId) -> u32 {
    table.get(&op).expect("effect op was not registered in the op table").tag
}

/// Extract the return type of an effect from a handler's MIR type. Handlers are shaped as
/// `fn op_args.. (resume: fn r1 [Pointer] -> r2) -> r2`, so an effect's return type r1 is
/// always the sole parameter of the resume parameter.
fn resume_arg_type_from_handler(handler_type: &Type) -> Option<Type> {
    let Type::Function(ft) = handler_type else { return None };
    let Type::Function(resume_ft) = ft.parameters.last()? else { return None };
    resume_ft.parameters.first().cloned()
}

fn size_of_const(typ: &Type, ptr_size: u32) -> Value {
    Value::Integer(IntConstant::Usz(typ.size_in_bytes(ptr_size) as usize))
}

fn is_zero_sized(typ: &Type) -> bool {
    match typ {
        Type::Primitive(PrimitiveType::Unit | PrimitiveType::NoClosureEnv) => true,
        Type::Tuple(fields) => fields.iter().all(is_zero_sized),
        _ => false,
    }
}

/// Placeholder value for stack slots that are immediately overwritten
/// (e.g. the destination of `mco_coro_pop`). Avoids [Value::Error] since
/// LLVM codegen rejects it.
/// TODO: Remove. Change StackAlloc to take a size and no initial value
fn dummy_value(typ: &Type) -> Value {
    match typ {
        Type::Primitive(primitive) => match primitive {
            PrimitiveType::Bool => Value::Bool(false),
            PrimitiveType::Char => Value::Char('\0'),
            PrimitiveType::Int(kind) => Value::Integer(match kind {
                IntegerKind::I8 => IntConstant::I8(0),
                IntegerKind::I16 => IntConstant::I16(0),
                IntegerKind::I32 => IntConstant::I32(0),
                IntegerKind::I64 => IntConstant::I64(0),
                IntegerKind::Isz => IntConstant::Isz(0),
                IntegerKind::U8 => IntConstant::U8(0),
                IntegerKind::U16 => IntConstant::U16(0),
                IntegerKind::U32 => IntConstant::U32(0),
                IntegerKind::U64 => IntConstant::U64(0),
                IntegerKind::Usz => IntConstant::Usz(0),
            }),
            _ => Value::Unit,
        },
        // Rest of the cases are broken
        _ => Value::Unit,
    }
}

enum EmitTarget<'local> {
    Block(BlockId),
    Pending(&'local mut Vec<InstructionId>),
}

struct Emitter<'local> {
    definition: &'local mut Definition,
    target: EmitTarget<'local>,
}

impl<'local> Emitter<'local> {
    fn in_block(definition: &'local mut Definition, block: BlockId) -> Self {
        Self { definition, target: EmitTarget::Block(block) }
    }

    fn pending(definition: &'local mut Definition, pending: &'local mut Vec<InstructionId>) -> Self {
        Self { definition, target: EmitTarget::Pending(pending) }
    }

    /// Push an instruction and return its result.
    fn push_instruction(&mut self, instruction: Instruction, result_type: Type) -> Value {
        let id = self.definition.instructions.push(instruction);
        self.definition.instruction_result_types.push_existing(id, result_type);
        self.append_instruction(id);
        Value::InstructionResult(id)
    }

    /// Repurpose an existing InstructionId's slot, appending it to the
    /// current target. Used when splicing new instructions over an original
    /// Handle/Perform slot so existing users of that slot's result continue
    /// to see the expected value.
    fn reuse_instruction(&mut self, id: InstructionId, instruction: Instruction, result_type: Type) -> Value {
        self.definition.instructions[id] = instruction;
        self.definition.instruction_result_types[id] = result_type;
        self.append_instruction(id);
        Value::InstructionResult(id)
    }

    fn append_instruction(&mut self, id: InstructionId) {
        match &mut self.target {
            EmitTarget::Block(block) => self.definition.blocks[*block].instructions.push(id),
            EmitTarget::Pending(pending) => pending.push(id),
        }
    }

    /// Emit `Extern(f)` + `Call(f, args)` and return the call result.
    fn call_extern(&mut self, function: &AminicoroFn, arguments: Vec<Value>) -> Value {
        let function_value =
            self.push_instruction(Instruction::Extern(function.name.to_string()), function.typ.clone());
        let Type::Function(function_type) = &function.typ else { panic!("McoFn.typ must be a Function") };
        let return_type = function_type.return_type.clone();
        self.push_instruction(Instruction::Call { function: function_value, arguments }, return_type)
    }

    /// Push a typed value onto a coroutine's channel via `mco_coro_push`.
    /// Zero-sized values are silently skipped (mco_coro_push rejects zero-length pushes).
    fn push_bytes(&mut self, mco: &AminicoroFns, coro: Value, value: Value, value_type: &Type, ptr_size: u32) {
        if is_zero_sized(value_type) {
            return;
        }
        if let Type::Tuple(fields) = value_type {
            for (index, field_type) in fields.iter().enumerate() {
                let field = self.push_instruction(
                    Instruction::IndexTuple { tuple: value, index: index as u32 },
                    field_type.clone(),
                );
                self.push_bytes(mco, coro, field, field_type, ptr_size);
            }
            return;
        }
        let slot = self.push_instruction(Instruction::StackAlloc(value), Type::POINTER);
        let size = size_of_const(value_type, ptr_size);
        self.call_extern(&mco.push, vec![coro, slot, size]);
    }

    /// Pop a typed value off a coroutine's channel. Returns `Value::Unit` for zero-sized types.
    fn pop_bytes(&mut self, mco: &AminicoroFns, coro: Value, value_type: &Type, ptr_size: u32) -> Value {
        if is_zero_sized(value_type) {
            return Value::Unit;
        }
        if let Type::Tuple(fields) = value_type {
            let mut popped = mapvec(fields.iter().rev(), |field_type| self.pop_bytes(mco, coro, field_type, ptr_size));
            popped.reverse();
            return self.push_instruction(Instruction::MakeTuple(popped), value_type.clone());
        }
        let slot = self.push_instruction(Instruction::StackAlloc(dummy_value(value_type)), Type::POINTER);
        let size = size_of_const(value_type, ptr_size);
        self.call_extern(&mco.pop, vec![coro, slot, size]);
        self.push_instruction(Instruction::Deref(slot), value_type.clone())
    }
}

/// Push `arguments` in declared order followed by `tag` onto `coro`.
///
/// aminicoro's channel is LIFO, so the tag is pushed last.
/// Per-case pops then consume args in reverse declaration order.
fn emit_push_arguments_and_tag(emitter: &mut Emitter, context: Context, coro: Value, arguments: &[Value], tag: u32) {
    for argument in arguments {
        let argument_type = emitter.definition.type_of_value(argument, &FxHashMap::default(), &FxHashMap::default());
        emitter.push_bytes(context.mco, coro, *argument, &argument_type, context.ptr_size);
    }
    emitter.push_bytes(
        context.mco,
        coro,
        Value::Integer(IntConstant::U32(tag)),
        &Type::int(IntegerKind::U32),
        context.ptr_size,
    );
}

/// `running = mco_coro_running (); push running args..; push running tag; suspend running`.
/// Shared by `Perform` lowering and forwarded-case lowering; both raise an
/// effect on whatever coroutine is currently running and let the enclosing
/// `drive` handle dispatch.
fn emit_raise_on_running(emitter: &mut Emitter, context: Context, arguments: &[Value], tag: u32) -> Value {
    let running = emitter.call_extern(&context.mco.running, vec![]);
    emit_push_arguments_and_tag(emitter, context, running, arguments, tag);
    emitter.call_extern(&context.mco.suspend, vec![running]);
    running
}

/// Pop a typed sequence of arguments off `coro`, restoring declared order.
/// aminicoro's channel is LIFO so args pushed last come off first; we pop
/// against the reversed type list, then flip the result back to declared order.
fn pop_operation_arguments(emitter: &mut Emitter, context: Context, coro: Value, arg_types: &[Type]) -> Vec<Value> {
    let mut popped = mapvec(arg_types.iter().rev(), |typ| emitter.pop_bytes(context.mco, coro, typ, context.ptr_size));
    popped.reverse();
    popped
}

/// Build the argument list for a `drive` call: `[coro, handlers..]`.
fn drive_call_arguments(coro: Value, handlers: impl IntoIterator<Item = Value>) -> Vec<Value> {
    std::iter::once(coro).chain(handlers).collect()
}

struct PerformSite {
    block: BlockId,
    index: usize,
    id: InstructionId,
    op: DefinitionId,
    arguments: Vec<Value>,
}

fn collect_perform_sites(definition: &Definition) -> Vec<PerformSite> {
    collect_sites(definition, |block, index, id| {
        if let Instruction::Perform { effect_op, arguments } = &definition.instructions[id] {
            Some(PerformSite { block, index, id, op: *effect_op, arguments: arguments.clone() })
        } else {
            None
        }
    })
}

fn rewrite_single_perform(mir: &mut Mir, definition_id: DefinitionId, site: PerformSite, context: Context) {
    let PerformSite { block, index, id: original_id, op, arguments } = site;

    // Read the Perform's return type directly from the instruction-result
    // map. Looking up `op`'s declared type via `Value::Definition(op)`
    // breaks when the effect op is generic and filtered out of the
    // monomorphized MIR (see regression 217_filter).
    let return_type = mir.definitions[&definition_id].instruction_result_types[original_id].clone();
    let tag = op_tag(context.op_table, op);
    let definition = mir.definitions.get_mut(&definition_id).expect("definition disappeared mid-rewrite");

    let mut pending = Vec::new();
    {
        let mut emitter = Emitter::pending(definition, &mut pending);
        emit_perform_sequence(&mut emitter, context, tag, &arguments, &return_type, original_id);
    }
    definition.blocks[block].instructions.splice(index..=index, pending);
}

/// Emit the instructions implementing a single `Perform`. The final Deref
/// (or `Id(Unit)` for zero-sized results) reuses `reuse_id` so existing
/// users referring to the Perform's result continue to work.
fn emit_perform_sequence(
    emitter: &mut Emitter, context: Context, tag: u32, arguments: &[Value], return_type: &Type, reuse_id: InstructionId,
) {
    let running = emit_raise_on_running(emitter, context, arguments, tag);

    let result = emitter.pop_bytes(context.mco, running, return_type, context.ptr_size);
    emitter.reuse_instruction(reuse_id, Instruction::Id(result), return_type.clone());
}

struct HandleSite {
    block: BlockId,
    index: usize,
    id: InstructionId,
    body: Value,
    cases: Vec<HandlerCase>,
    result_type: Type,
}

fn collect_handle_sites(definition: &Definition) -> Vec<HandleSite> {
    collect_sites(definition, |block, index, id| {
        if let Instruction::Handle { body, cases } = &definition.instructions[id] {
            Some(HandleSite {
                block,
                index,
                id,
                body: *body,
                cases: cases.clone(),
                result_type: definition.instruction_result_types[id].clone(),
            })
        } else {
            None
        }
    })
}

fn rewrite_single_handle(mir: &mut Mir, definition_id: DefinitionId, site: HandleSite, context: Context) {
    let HandleSite { block, index, id: original_id, body, cases, result_type } = site;

    let definition = mir.definitions.get(&definition_id).expect("definition vanished mid-rewrite");
    let body_type = definition.type_of_value(&body, &mir.externals, &mir.definitions);
    let handler_types =
        mapvec(&cases, |case| definition.type_of_value(&case.handler, &mir.externals, &mir.definitions));

    let body_wrapper_id = generate_body_wrapper(mir, body_type, &result_type, context);
    let drive_id = generate_drive_function(mir, &cases, &handler_types, &result_type, context);

    let definition = mir.definitions.get_mut(&definition_id).expect("definition disappeared mid-rewrite");
    let mut pending = Vec::new();

    let mut emitter = Emitter::pending(definition, &mut pending);

    let body_slot = emitter.push_instruction(Instruction::StackAlloc(body), Type::POINTER);

    // Function values are pointers at the LLVM level but MIR distinguishes them.
    // Transmute to make the call type-check.
    let wrapper_ptr =
        emitter.push_instruction(Instruction::Transmute(Value::Definition(body_wrapper_id)), Type::POINTER);

    let coro = emitter.call_extern(&context.mco.init, vec![wrapper_ptr, body_slot]);

    // Drive returns `result_type` directly. Reuse the original Handle's
    // InstructionId so existing users of the Handle's value keep working.
    let drive_arguments = drive_call_arguments(coro, cases.iter().map(|case| case.handler));
    emitter.reuse_instruction(
        original_id,
        Instruction::Call { function: Value::Definition(drive_id), arguments: drive_arguments },
        result_type.clone(),
    );

    emitter.call_extern(&context.mco.free, vec![coro]);
    definition.blocks[block].instructions.splice(index..=index, pending);
}

/// Generate `fn (coro: Pointer) -> Unit` that reads the body closure from the
/// coroutine's user_data, invokes it, and pushes the result onto the
/// coroutine's channel for the trampoline to pop.
fn generate_body_wrapper(mir: &mut Mir, body_type: Type, result_type: &Type, context: Context) -> DefinitionId {
    let wrapper_id = next_definition_id();
    let wrapper_type = ptr_fn(vec![Type::POINTER], Type::UNIT);

    let mut definition = Definition::new(Arc::new("handle_body_wrapper".to_string()), wrapper_id, 0, wrapper_type);
    let entry = BlockId::ENTRY_BLOCK;
    definition.blocks[entry].parameter_types.push(Type::POINTER);
    let coro = Value::Parameter(entry, 0);

    let mut emitter = Emitter::in_block(&mut definition, entry);

    let user_data = emitter.call_extern(&context.mco.get_user_data, vec![coro]);
    let body_value = emitter.push_instruction(Instruction::Deref(user_data), body_type.clone());

    // The body may be a closure tuple (fn, env) or a plain function.
    let call = match &body_type {
        Type::Function(function_type) if function_type.is_closure() => {
            Instruction::CallClosure { closure: body_value, arguments: vec![] }
        },
        Type::Tuple(_) => Instruction::CallClosure { closure: body_value, arguments: vec![] },
        _ => Instruction::Call { function: body_value, arguments: vec![] },
    };
    let result = emitter.push_instruction(call, result_type.clone());
    emitter.push_bytes(context.mco, coro, result, result_type, context.ptr_size);

    definition.blocks[entry].terminator = Some(TerminatorInstruction::Return(Value::Unit));
    mir.definitions.insert(wrapper_id, definition);
    wrapper_id
}

/// Generate the drive function for a specific Handle site:
/// `drive (coro: Pointer) handlers.. -> r2`.
///
/// The generated function resumes the coroutine once, then inspects whether
/// the body completed or suspended. On completion it pops and returns r2.
/// On suspension it pops the op_tag and dispatches to the matching handler
/// case, passing a per-case resume closure that re-enters `drive` once the
/// coroutine is pushed the resumed value.
fn generate_drive_function(
    mir: &mut Mir, cases: &[HandlerCase], handler_types: &[Type], result_type: &Type, context: Context,
) -> DefinitionId {
    let drive_id = next_definition_id();
    let mut drive_parameters = vec![Type::POINTER];
    drive_parameters.extend(handler_types.iter().cloned());
    let drive_type = ptr_fn(drive_parameters.clone(), result_type.clone());

    // One resume helper per case. Each captures (coro, handlers..) as its
    // env so it can re-invoke drive with the same handler arguments.
    let resume_functions = mapvec(handler_types, |handler_type| {
        let arg_type = resume_arg_type_from_handler(handler_type).unwrap_or(Type::UNIT);
        generate_resume_function(mir, arg_type, drive_id, handler_types, result_type, context)
    });

    // Drive needs a Switch case for *every* op in the program, not just
    // those it handles — forwarded ops need cases too so we can keep the
    // typed push/pop layout.
    let handled_ops: FxHashMap<DefinitionId, usize> =
        cases.iter().enumerate().map(|(index, case)| (case.effect_op, index)).collect();

    let mut definition = Definition::new(Arc::new("handle_drive".to_string()), drive_id, 0, drive_type);
    let entry = BlockId::ENTRY_BLOCK;
    for parameter_type in &drive_parameters {
        definition.blocks[entry].parameter_types.push(parameter_type.clone());
    }
    let coro = Value::Parameter(entry, 0);
    let handler_parameters = mapvec(0..handler_types.len(), |i| Value::Parameter(entry, (i + 1) as u32));

    // All control-flow paths converge at `final_block (r: R)` which simply
    // returns r. This shape keeps topological_sort happy because each
    // branch terminates with a Jmp to a common merge point rather than
    // inline Returns.
    let dispatch_block = definition.blocks.push(Block::new(Vec::new()));
    let complete_block = definition.blocks.push(Block::new(Vec::new()));
    let final_block = definition.blocks.push(Block::new(vec![result_type.clone()]));

    let mut emitter = Emitter::in_block(&mut definition, entry);
    emitter.call_extern(&context.mco.resume, vec![coro]);
    let suspended = emitter.call_extern(&context.mco.is_suspended, vec![coro]);

    // If `end == else_` here, `topological_sort` does not push a merge
    // point for this If. The actual convergence happens at `final_block`
    // via Jmps from both the Switch cases and `complete_block`; the If
    // itself doesn't need to register a merge, so we use `else_.0` as the end.
    definition.blocks[entry].terminator = Some(TerminatorInstruction::If {
        condition: suspended,
        then: (dispatch_block, None),
        else_: (complete_block, None),
        end: complete_block,
    });

    // complete_block: pop R, jmp final_block(R)
    emit_pop_and_jmp(&mut definition, complete_block, coro, result_type, final_block, context);

    // dispatch_block: pop u32 tag, switch to case block
    let mut emitter = Emitter::in_block(&mut definition, dispatch_block);
    let tag = emitter.pop_bytes(context.mco, coro, &Type::int(IntegerKind::U32), context.ptr_size);

    let mut case_blocks = Vec::with_capacity(context.op_table.len());
    let mut op_entries = context.op_table.iter().collect::<Vec<_>>();
    op_entries.sort_by_key(|(_, info)| info.tag);

    for (op_id, info) in op_entries {
        let case_block = definition.blocks.push(Block::new(Vec::new()));
        case_blocks.push((info.tag, (case_block, None)));

        if let Some(&case_index) = handled_ops.get(op_id) {
            emit_handler_case(
                &mut definition,
                case_block,
                coro,
                resume_functions[case_index],
                handler_parameters[case_index],
                &handler_parameters,
                result_type,
                final_block,
                mir,
                context,
            );
        } else if let Some(signature) = info.signature.as_ref() {
            emit_forward_case(
                &mut definition,
                case_block,
                coro,
                info.tag,
                signature,
                drive_id,
                &handler_parameters,
                result_type,
                final_block,
                context,
            );
        } else {
            // Op has a tag but no known signature — only possible when it's
            // referenced by a Perform but never handled anywhere. Performing
            // it was already UB; make it an unreachable case here so codegen
            // doesn't complain about block parameters.
            definition.blocks[case_block].terminator = Some(TerminatorInstruction::Unreachable);
        }
    }

    let unreachable_else = definition.blocks.push(Block::new(Vec::new()));
    definition.blocks[unreachable_else].terminator = Some(TerminatorInstruction::Unreachable);

    definition.blocks[dispatch_block].terminator = Some(TerminatorInstruction::Switch {
        int_value: tag,
        cases: case_blocks,
        else_: Some((unreachable_else, None)),
        end: final_block,
    });

    definition.blocks[final_block].terminator = Some(TerminatorInstruction::Return(Value::Parameter(final_block, 0)));

    mir.definitions.insert(drive_id, definition);
    drive_id
}

/// In `block`: pop a value of `value_type` off `coro` and jmp to `jmp_target` with the popped
/// value as the jump argument.
fn emit_pop_and_jmp(
    definition: &mut Definition, block: BlockId, coro: Value, value_type: &Type, jmp_target: BlockId, context: Context,
) {
    let value = {
        let mut emitter = Emitter::in_block(definition, block);
        emitter.pop_bytes(context.mco, coro, value_type, context.ptr_size)
    };
    definition.blocks[block].terminator = Some(TerminatorInstruction::Jmp((jmp_target, Some(value))));
}

/// Emit the per-case dispatch body: pop the op's arguments, build the resume closure, invoke the
/// user-supplied handler, and jump to `final_block` with its result.
#[allow(clippy::too_many_arguments)]
fn emit_handler_case(
    definition: &mut Definition, case_block: BlockId, coro: Value, resume_function_id: DefinitionId,
    handler_parameter: Value, all_handler_parameters: &[Value], result_type: &Type, final_block: BlockId, mir: &Mir,
    context: Context,
) {
    // Derive the op's parameter types from the handler's signature. Handler shape:
    // `fn op_args.., resume -> r2`. This avoids looking up the effect op's DefinitionId, which may
    // be unavailable after monomorphization filters out generic definitions (regression 217_filter).
    let handler_type = definition.type_of_value(&handler_parameter, &mir.externals, &mir.definitions);
    let Type::Function(handler_function_type) = &handler_type else { return };
    let op_parameter_types =
        handler_function_type.parameters.split_last().map(|(_, rest)| rest.to_vec()).unwrap_or_default();
    let handler_is_closure = handler_function_type.is_closure();

    let resume_function_type = mir.definitions.get(&resume_function_id).map(|d| d.typ.clone()).unwrap_or(Type::ERROR);

    let parameters = all_handler_parameters.iter();
    let state_field_types = std::iter::once(Type::POINTER)
        .chain(parameters.map(|value| definition.type_of_value(value, &mir.externals, &mir.definitions)))
        .collect::<Vec<_>>();

    let mut emitter = Emitter::in_block(definition, case_block);
    let popped_arguments = pop_operation_arguments(&mut emitter, context, coro, &op_parameter_types);

    // Build the resume closure's environment: pack (coro, handlers..)
    // into an inline tuple and stack-allocate it. The MIR builder gave
    // `resume` env type `Pointer`, so we pass a pointer to this tuple.
    let state_elements = drive_call_arguments(coro, all_handler_parameters.iter().copied());
    let state =
        emitter.push_instruction(Instruction::MakeTuple(state_elements), Type::Tuple(Arc::new(state_field_types)));
    let environment = emitter.push_instruction(Instruction::StackAlloc(state), Type::POINTER);

    let resume_closure = emitter.push_instruction(
        Instruction::PackClosure { function: Value::Definition(resume_function_id), environment },
        resume_function_type,
    );

    // Call the handler: handler(args.., resume_closure)
    let mut handler_arguments = popped_arguments;
    handler_arguments.push(resume_closure);
    let call_instruction = if handler_is_closure {
        Instruction::CallClosure { closure: handler_parameter, arguments: handler_arguments }
    } else {
        Instruction::Call { function: handler_parameter, arguments: handler_arguments }
    };
    let handler_result = emitter.push_instruction(call_instruction, result_type.clone());

    definition.blocks[case_block].terminator = Some(TerminatorInstruction::Jmp((final_block, Some(handler_result))));
}

/// Emit a case body that forwards an op this Handle doesn't handle up to
/// the outer enclosing drive. The inner coro is currently suspended, so
/// `mco_coro_running()` yields the outer. We re-push the op's args + tag
/// onto the outer in the same layout `Perform` uses, suspend the outer,
/// receive its resumed V, push V back to the inner, and recurse into drive.
#[allow(clippy::too_many_arguments)]
fn emit_forward_case(
    definition: &mut Definition, case_block: BlockId, inner_coro: Value, tag: u32, signature: &EffectSignature,
    drive_id: DefinitionId, handler_parameters: &[Value], result_type: &Type, final_block: BlockId, context: Context,
) {
    let mut emitter = Emitter::in_block(definition, case_block);

    let popped_arguments = pop_operation_arguments(&mut emitter, context, inner_coro, &signature.arg_types);
    let outer_coro = emit_raise_on_running(&mut emitter, context, &popped_arguments, tag);

    // Pop resumed off outer, push it back to inner, then recur so the inner keeps running.
    let resume_arg = emitter.pop_bytes(context.mco, outer_coro, &signature.return_type, context.ptr_size);
    emitter.push_bytes(context.mco, inner_coro, resume_arg, &signature.return_type, context.ptr_size);

    let drive_arguments = drive_call_arguments(inner_coro, handler_parameters.iter().copied());
    let recurse_result = emitter.push_instruction(
        Instruction::Call { function: Value::Definition(drive_id), arguments: drive_arguments },
        result_type.clone(),
    );

    definition.blocks[case_block].terminator = Some(TerminatorInstruction::Jmp((final_block, Some(recurse_result))));
}

/// Generate a resume closure specialized for a particular Handle site:
/// `resume: fn r1 (env: Pointer) -> r2`. `env` points to a stack-allocated
/// tuple `(coro, handler0, handler1, ...)` set up by the drive function.
fn generate_resume_function(
    mir: &mut Mir, r1_type: Type, drive_function_id: DefinitionId, handler_types: &[Type], result_type: &Type,
    context: Context,
) -> DefinitionId {
    let mut state_field_types = Vec::with_capacity(1 + handler_types.len());
    state_field_types.push(Type::POINTER);
    state_field_types.extend(handler_types.iter().cloned());
    let state_tuple_type = Type::Tuple(Arc::new(state_field_types));

    let function_type = Type::Function(Arc::new(FunctionType {
        parameters: vec![r1_type.clone()],
        environment: Type::POINTER,
        return_type: result_type.clone(),
    }));

    let resume_id = next_definition_id();
    let mut definition = Definition::new(Arc::new("handle_resume".to_string()), resume_id, 0, function_type);
    let entry = BlockId::ENTRY_BLOCK;
    definition.blocks[entry].parameter_types.push(r1_type.clone());
    definition.blocks[entry].parameter_types.push(Type::POINTER);
    let v_value = Value::Parameter(entry, 0);
    let environment_pointer = Value::Parameter(entry, 1);

    let mut emitter = Emitter::in_block(&mut definition, entry);

    let state = emitter.push_instruction(Instruction::Deref(environment_pointer), state_tuple_type);
    let coro = emitter.push_instruction(Instruction::IndexTuple { tuple: state, index: 0 }, Type::POINTER);
    let handler_values = mapvec(handler_types.iter().enumerate(), |(i, handler_type)| {
        emitter.push_instruction(Instruction::IndexTuple { tuple: state, index: (i + 1) as u32 }, handler_type.clone())
    });

    emitter.push_bytes(context.mco, coro, v_value, &r1_type, context.ptr_size);

    // Recursively call drive to continue executing the coroutine
    // through further suspensions. Pass the full handler set so drive
    // can dispatch further Performs.
    let drive_arguments = drive_call_arguments(coro, handler_values);
    let drive_result = emitter.push_instruction(
        Instruction::Call { function: Value::Definition(drive_function_id), arguments: drive_arguments },
        result_type.clone(),
    );

    definition.blocks[entry].terminator = Some(TerminatorInstruction::Return(drive_result));

    mir.definitions.insert(resume_id, definition);
    resume_id
}
