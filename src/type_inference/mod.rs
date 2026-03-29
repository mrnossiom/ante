use std::{collections::BTreeMap, rc::Rc, sync::Arc};

use inc_complete::DbGet;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use crate::{
    diagnostics::{Diagnostic, Location},
    incremental::{self, DbHandle, GetItem, Resolve, TargetPointerSize, TypeCheck, TypeCheckSCC},
    iterator_extensions::mapvec,
    lexer::token::IntegerKind,
    name_resolution::{Origin, ResolutionResult},
    parser::{
        context::TopLevelContext,
        cst::{self, Name, ReferenceKind, TopLevelItem, TopLevelItemKind},
        ids::{ExprId, NameId, PathId, PatternId, TopLevelId, TopLevelName},
    },
    type_inference::{
        dependency_graph::TypeCheckResult,
        errors::{Locateable, TypeErrorKind},
        fresh_expr::ExtendedTopLevelContext,
        generics::Generic,
        types::{PrimitiveType, Type, TypeBindings, TypeVariableId},
    },
};

mod cst_traversal;
pub mod dependency_graph;
pub mod errors;
mod free_variables;
pub mod fresh_expr;
pub mod generics;
mod get_type;
mod implicits;
pub mod kinds;
pub mod patterns;
mod type_definitions;
pub mod types;

pub use get_type::get_type_impl;

/// Actually type check a statement and its contents.
/// Unlike `get_type_impl`, this always type checks the expressions inside a statement
/// to ensure they type check correctly.
pub fn type_check_impl(context: &TypeCheckSCC, compiler: &DbHandle) -> TypeCheckSCCResult {
    incremental::enter_query();
    let items = TypeChecker::item_contexts(&context.0, compiler);
    let mut checker = TypeChecker::new(&items, compiler);

    let items = mapvec(context.0.iter(), |item_id| {
        incremental::println(format!("Type checking {item_id:?}"));
        checker.start_item(*item_id);
        checker.push_implicits_scope();

        let item = &checker.item_contexts[item_id].0;
        match &item.kind {
            TopLevelItemKind::Definition(definition) => checker.check_definition(definition),
            TopLevelItemKind::TypeDefinition(type_definition) => checker.check_type_definition(type_definition),
            TopLevelItemKind::TraitDefinition(_) => unreachable!("Traits should be desugared into types by this point"),
            TopLevelItemKind::TraitImpl(_) => unreachable!("Impls should be simplified into definitions by this point"),
            TopLevelItemKind::EffectDefinition(_) => (), // TODO
            TopLevelItemKind::Extern(extern_) => checker.check_extern(extern_),
            TopLevelItemKind::Comptime(comptime) => checker.check_comptime(comptime),
        };

        checker.pop_implicits_scope();
        (*item_id, checker.finish_item())
    });

    incremental::exit_query();
    checker.finish(items)
}

/// A `TypeCheckSCCResult` holds the `IndividualTypeCheckResult` of every item in
/// the SCC for a particular TopLevelId
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeCheckSCCResult {
    pub items: BTreeMap<TopLevelId, IndividualTypeCheckResult>,
    pub bindings: TypeBindings,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndividualTypeCheckResult {
    #[serde(flatten)]
    pub maps: TypeMaps,

    /// The type checker may create additional expressions, patterns, etc.,
    /// which it places in this context. This is a full replacement for the
    /// [TopLevelContext] output from the parser. Continuing to use the old
    /// [TopLevelContext] will work for most expressions but lead to panics
    /// when newly created items from the type checking pass are used.
    pub context: ExtendedTopLevelContext,

    /// One or more names may be externally visible outside this top-level item.
    /// Each of these names will be generalized and placed in this map.
    /// Ex: in `foo = (bar = 1; bar + 2)` only `foo: I32` will be generalized,
    /// but in `a, b = 1, 2`, both `a` and `b` will be.
    /// Ex2: in `type Foo = | A | B`, `A` and `B` will both be generalized, and
    /// there is no need to generalize `Foo` itself.
    pub generalized: BTreeMap<NameId, Type>,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TypeMaps {
    pub name_types: BTreeMap<NameId, Type>,
    pub path_types: BTreeMap<PathId, Type>,
    pub expr_types: BTreeMap<ExprId, Type>,
    pub pattern_types: BTreeMap<PatternId, Type>,
}

/// The TypeChecker is responsible for checking for type errors inside of an
/// inference group. An inference group is a set of top-level items which form
/// an SCC in the type inference dependency graph. Usually each group is only
/// a single item but larger groups are possible for mutually recursive definitions
/// without type signatures.
///
/// The TypeChecker is the main context object for the type inference incremental computation.
/// Its outputs are:
/// - A type for all [NameId], [PathId], and [ExprId] objects (possibly an error type)
/// - Errors or warnings accumulated to the compiler's [Diagnostic] list
/// - A new resolved [Origin] for each [Origin::TypeResolution] outputted from the name resolution pass
/// - New expressions & paths resulting from the compilation of match expressions into decision trees
struct TypeChecker<'local, 'inner> {
    compiler: &'local DbHandle<'inner>,
    name_types: BTreeMap<NameId, Type>,
    path_types: BTreeMap<PathId, Type>,
    pattern_types: BTreeMap<PatternId, Type>,
    expr_types: BTreeMap<ExprId, Type>,

    bindings: TypeBindings,

    /// Type inference is the first pass where type variables are introduced.
    /// This field starts from 0 to give each a unique ID within the current inference group.
    next_type_variable_id: u32,

    /// Contains the ItemContext for each item in the TypeChecker's type check group.
    /// Most often, this is just a single item. In the case of mutually recursive type
    /// inference however, it will include every item in the recursive SCC to infer.
    item_contexts: &'local ItemContexts,

    /// The type checker may output new expression, path, or name IDs so we
    /// extend each [TopLevelContext] with these new ids.
    id_contexts: FxHashMap<TopLevelId, ExtendedTopLevelContext>,

    /// The current top-level item being type checked. This is empty upon initialization, but
    /// while type checking, this should always be non-empty.
    current_item: Option<TopLevelId>,

    /// The return type of the current function. Used to type check `return` statements.
    function_return_type: Option<Type>,

    /// Types of each top-level item in the current SCC being worked on
    item_types: Rc<FxHashMap<TopLevelName, Type>>,

    /// The outer Vec represents each scope (roughly each block of code),
    /// while the inner Vec is the implicits context for that scope. This contains
    implicits: Vec<ImplicitsContext>,

    /// Tracks ExprIds for which `check_lambda` was called due to an implicit parameter coercion
    /// wrapper. For these, `check_for_closure` is deferred until after `pop_implicits_scope` of
    /// the enclosing lambda resolves the delayed implicits that fill in the wrapper's free vars.
    coercion_wrapper_exprs: FxHashSet<ExprId>,
}

#[derive(Default, Clone)]
struct ImplicitsContext {
    /// Any implicits introduced in the current scope. To find all implicits in scope, it is
    /// necessary to traverse all levels of `TypeChecker::implicits`, in addition to querying
    /// implicits in global scope separately.
    implicits_in_scope: Vec<NameId>,

    /// Contains implicits for which we need to delay checking for a value for until the end of the
    /// current item when more types are inferred. Without this, for example, we'd see `0i32 < 3`
    /// and would fail searching for an implicit for `Cmp _` since we'd check `<` before its
    /// arguments while its type is still unknown.
    delayed_implicits: Vec<DelayedImplicit>,

    /// Closure checks deferred for coercion wrapper lambdas. These are run after `delayed_implicits`
    /// are resolved so that free-variable analysis sees the fully-resolved implicit arguments.
    deferred_closure_checks: Vec<(ExprId, Type)>,

    /// Any type variables created for integer literals for polymorphic integer types.
    /// If not bound by the end of a scope they will be defaulted to I32.
    /// This is a tuple of (the integer's value, the integer type variable, location to use for errors)
    integer_type_variables: Vec<(u64, TypeVariableId, Location)>,
}

#[derive(Clone, Copy)]
struct DelayedImplicit {
    /// The [ExprId] which originally requested an implicit value.
    /// This is often a trait function like `cast` or `+`
    source: ExprId,

    /// The destination to emplace the implicit value into
    destination: ExprId,

    /// The parameter index the implicit should slot into on the `self.source` expr.
    /// Used in error messages.
    parameter_index: usize,
}

/// Map from each TopLevelId to a tuple of (the item, parse context, resolution context)
type ItemContexts = FxHashMap<TopLevelId, (Arc<TopLevelItem>, Arc<TopLevelContext>, ResolutionResult)>;

impl<'local, 'inner> TypeChecker<'local, 'inner> {
    fn new(item_contexts: &'local ItemContexts, compiler: &'local DbHandle<'inner>) -> Self {
        let id_contexts = item_contexts
            .iter()
            .map(|(id, (_, context, _))| (*id, ExtendedTopLevelContext::new(context.clone())))
            .collect();

        let mut this = Self {
            compiler,
            bindings: Default::default(),
            next_type_variable_id: 0,
            name_types: Default::default(),
            path_types: Default::default(),
            expr_types: Default::default(),
            pattern_types: Default::default(),
            item_types: Default::default(),
            current_item: None,
            function_return_type: None,
            item_contexts,
            id_contexts,
            implicits: vec![Default::default()],
            coercion_wrapper_exprs: Default::default(),
        };

        let mut item_types = FxHashMap::default();
        for (item_id, (_, _, resolution)) in item_contexts.iter() {
            for name in resolution.top_level_names.iter() {
                let variable = this.next_type_variable();
                item_types.insert(TopLevelName::new(*item_id, *name), variable);
            }
        }
        // We have to go through this extra step since `generalize_all` needs an Rc
        // to clone this field cheaply since `generalize` requires a mutable `self`.
        let this_item_types = Rc::get_mut(&mut this.item_types).expect("No clones should be possible here");
        *this_item_types = item_types;

        this
    }

    fn item_contexts(items: &[TopLevelId], compiler: &DbHandle) -> ItemContexts {
        items
            .iter()
            .map(|item_id| {
                let (item, item_context) = GetItem(*item_id).get(compiler);
                let resolve = Resolve(*item_id).get(compiler);
                (*item_id, (item, item_context, resolve))
            })
            .collect()
    }

    /// Returns the context of the current item, containing mappings for IDs set during parsing.
    /// This will not contain any new IDs added by this type checking pass - for that use
    /// [Self::current_extended_context_mut]. This method is still useful since the returned
    /// context refers to a separate lifetime, so self may still be used mutably.
    fn current_context(&self) -> &'local TopLevelContext {
        let item = self.current_item.expect("TypeChecker: Expected current_item to be set");
        &self.item_contexts[&item].1
    }

    fn current_resolve(&self) -> &'local ResolutionResult {
        let item = self.current_item.expect("TypeChecker: Expected current_item to be set");
        &self.item_contexts[&item].2
    }

    /// Return the current extended context.
    /// Note that this context only includes new items added by this type checker, it does
    /// not contain any existing items from the resolver until the type checker finishes
    /// and inserts the pre-existing items.
    fn current_extended_context(&self) -> &ExtendedTopLevelContext {
        let item = self.current_item.expect("TypeChecker: Expected current_item to be set");
        self.id_contexts.get(&item).expect("Expected TopLevelId to be in id_contexts")
    }

    /// Return the current extended context.
    /// Note that this context only includes new items added by this type checker, it does
    /// not contain any existing items from the resolver until the type checker finishes
    /// and inserts the pre-existing items.
    fn current_extended_context_mut(&mut self) -> &mut ExtendedTopLevelContext {
        let item = self.current_item.expect("TypeChecker: Expected current_item to be set");
        self.id_contexts.get_mut(&item).expect("Expected TopLevelId to be in id_contexts")
    }

    /// Returns the [Origin] of the given [PathId]. May return [None] if there
    /// was an error during name resolution.
    fn path_origin(&self, path: PathId) -> Option<Origin> {
        let origin = self.current_resolve().path_origins.get(&path).copied();
        origin.or_else(|| self.current_extended_context().path_origin(path))
    }

    fn finish(mut self, items: Vec<(TopLevelId, TypeMaps)>) -> TypeCheckSCCResult {
        let mut generalized = self.generalize_all();
        let items = items
            .into_iter()
            .map(|(id, maps)| {
                let generalized = generalized.remove(&id).unwrap_or_default();
                let mut context = self.id_contexts.remove(&id).unwrap();
                let item_context = self.item_contexts.get(&id).unwrap();
                context.extend_from_resolution_result(&item_context.2);
                (id, IndividualTypeCheckResult { maps, generalized, context })
            })
            .collect();

        TypeCheckSCCResult { items, bindings: self.bindings }
    }

    /// Check if the integer fits in the given kind, error if not
    fn check_int_fits(&self, value: u64, kind: IntegerKind, locator: impl Locateable) {
        let ptr_size = TargetPointerSize.get(self.compiler);
        let bit_size = 8 * kind.size_in_bytes(ptr_size);
        if bit_size == 64 {
            return;
        }

        // TODO: Change `value` repr from u64 to a type that fits negatives
        // so we can give more accurate ranges. As-is, u64::MAX fits into i64.
        if value > 2u64.pow(bit_size) - 1 {
            let location = locator.locate(self);
            self.compiler.accumulate(Diagnostic::IntegerTooLarge { value, kind, location });
        }
    }

    /// Prepare the TypeChecker to type check another item.
    fn start_item(&mut self, item_id: TopLevelId) {
        self.current_item = Some(item_id);

        // Iterating over every item type here should be fine for performance.
        // The expected length of `self.item_types` is 1 in the vast majority of cases,
        // and is only a bit longer with mutually recursive type-inferred definitions
        // and definitions defining multiple names (e.g. `a, b = 1, 2`).
        for (name, typ) in self.item_types.iter() {
            if name.top_level_item == item_id {
                self.name_types.insert(name.local_name_id, typ.clone());
            }
        }
    }

    /// Finishes the current item, adding all bindings to the relevant entry in
    /// `self.finished_items`, clearing them out in preparation for resolving the next item.
    fn finish_item(&mut self) -> TypeMaps {
        self.current_item = None;
        TypeMaps {
            name_types: std::mem::take(&mut self.name_types),
            path_types: std::mem::take(&mut self.path_types),
            expr_types: std::mem::take(&mut self.expr_types),
            pattern_types: std::mem::take(&mut self.pattern_types),
        }
    }

    fn next_type_variable_id(&mut self) -> TypeVariableId {
        let id = TypeVariableId(self.next_type_variable_id);
        self.next_type_variable_id += 1;
        id
    }

    fn next_type_variable(&mut self) -> Type {
        Type::Variable(self.next_type_variable_id())
    }

    /// Generalize all types in the current SCC.
    /// The returned Vec is in the same order as the SCC.
    ///
    /// Note that NameIds and PatternIds locally within each function will still refer to the
    /// non-generalized version of their types. If you want to retrieve the generalized type of an
    /// item from this SCC, you'll need to go through the generalized results specifically.
    fn generalize_all(&mut self) -> BTreeMap<TopLevelId, BTreeMap<NameId, Type>> {
        let mut items: BTreeMap<_, BTreeMap<_, _>> = BTreeMap::new();

        for (name, typ) in self.item_types.clone().iter() {
            self.current_item = Some(name.top_level_item);
            let typ = typ.generalize(&self.bindings);
            items.entry(name.top_level_item).or_default().insert(name.local_name_id, typ);
        }

        items
    }

    /// Unifies the two types. Returns false on failure
    fn unify(&mut self, actual: &Type, expected: &Type, kind: TypeErrorKind, locator: impl Locateable) -> bool {
        if self.try_unify(actual, expected).is_err() {
            let actual = self.type_to_string(actual);
            let expected = self.type_to_string(expected);
            let location = locator.locate(self);
            self.compiler.accumulate(Diagnostic::TypeError { actual, expected, kind, location });
            false
        } else {
            true
        }
    }

    /// Try to apply a coercion between `actual` and `expected`, returning a new expression
    /// if successful.
    ///
    /// Possible coercions:
    /// - If `actual` is a function type with more implicit parameters than `expected` has,
    /// search for implicit values in scope and create a new wrapper function over `expr`.
    ///
    /// Returns `true` if `expr` was modified, or `false` otherwise
    fn try_coercion(&mut self, actual: &Type, expected: &Type, expr: ExprId) -> bool {
        match (self.follow_type(actual), self.follow_type(expected)) {
            (Type::Function(actual_fn), Type::Function(expected_fn))
                if actual_fn.parameters.len() != expected_fn.parameters.len() =>
            {
                if let Some(new_expr) = self.implicit_parameter_coercion(actual_fn.clone(), expected_fn.clone(), expr) {
                    self.current_extended_context_mut().insert_expr(expr, new_expr);
                    self.coercion_wrapper_exprs.insert(expr);
                    true
                } else {
                    false
                }
            },
            _ => false,
        }
    }

    fn type_to_string(&self, typ: &Type) -> String {
        typ.to_string(&self.bindings, &self.current_context().names, self.compiler)
    }

    /// Try to unify the given types, returning `Err(())` on error without pushing a Diagnostic.
    ///
    /// Note that any type variable bindings will remain bound.
    fn try_unify(&mut self, actual: &Type, expected: &Type) -> Result<(), ()> {
        if actual == expected {
            return Ok(());
        }

        match (actual, expected) {
            (Type::Variable(actual_id), expected) => {
                if let Some(actual) = self.bindings.get(actual_id).cloned() {
                    self.try_unify(&actual, &expected)
                } else {
                    let expected = expected.follow(&self.bindings).clone();
                    self.try_bind_type_variable(*actual_id, expected)
                }
            },
            (actual, Type::Variable(expected_id)) => {
                if let Some(expected) = self.bindings.get(expected_id).cloned() {
                    self.try_unify(actual, &expected)
                } else {
                    let actual = actual.follow(&self.bindings).clone();
                    self.try_bind_type_variable(*expected_id, actual)
                }
            },
            (Type::Primitive(PrimitiveType::Error), _) | (_, Type::Primitive(PrimitiveType::Error)) => Ok(()),
            (Type::Function(actual), Type::Function(expected)) => {
                if actual.parameters.len() != expected.parameters.len() {
                    return Err(());
                }

                for (actual, expected) in actual.parameters.iter().zip(expected.parameters.iter()) {
                    if actual.is_implicit != expected.is_implicit {
                        return Err(());
                    }
                    self.try_unify(&actual.typ, &expected.typ)?;
                }

                self.try_unify(&actual.effects, &expected.effects)?;
                self.try_unify(&actual.return_type, &expected.return_type)
            },
            (
                Type::Application(actual_constructor, actual_args),
                Type::Application(expected_constructor, expected_args),
            ) => {
                if actual_args.len() != expected_args.len() {
                    return Err(());
                }
                self.try_unify(actual_constructor, expected_constructor)?;
                for (actual, expected) in actual_args.iter().zip(expected_args.iter()) {
                    self.try_unify(actual, expected)?;
                }
                Ok(())
            },
            (Type::Forall(actual_generics, actual), Type::Forall(expected_generics, expected)) => {
                if actual_generics.len() != expected_generics.len() {
                    return Err(());
                }
                for (actual, expected) in actual_generics.iter().zip(expected_generics.iter()) {
                    self.try_unify(&actual.as_type(), &expected.as_type())?;
                }
                self.try_unify(actual, expected)
            },
            (
                Type::Primitive(PrimitiveType::Reference(actual)),
                Type::Primitive(PrimitiveType::Reference(expected)),
            ) => {
                // Allow coercions between reference kinds: any ref type coerces to `ref`,
                // and `uniq` also coerces to `mut`.
                match (actual, expected) {
                    (_, ReferenceKind::Ref) => Ok(()),
                    (ReferenceKind::Uniq, ReferenceKind::Mut) => Ok(()),
                    (actual, expected) if actual == expected => Ok(()),
                    _ => Err(()),
                }
            },
            (actual, other) if actual == other => Ok(()),
            _ => Err(()),
        }
    }

    /// Try to bind a type variable, possibly erroring instead if the binding would lead
    /// to a recursive type.
    ///
    /// Before calling this function its argument must be zonked! `binding == binding.follow(...)`
    fn try_bind_type_variable(&mut self, id: TypeVariableId, binding: Type) -> Result<(), ()> {
        if binding == Type::Variable(id) {
            // Already equal, don't recursively bind self to self
            Ok(())
        } else if self.occurs(&binding, id) {
            // Recursive type error
            Err(())
        } else {
            self.bindings.insert(id, binding);
            Ok(())
        }
    }

    /// True if `variable` occurs within `typ`.
    /// Used to prevent the creation of infinitely recursive types when binding type variables.
    fn occurs(&self, typ: &Type, variable: TypeVariableId) -> bool {
        match typ {
            Type::Primitive(_) | Type::Generic(_) | Type::UserDefined(_) => false,
            Type::Variable(candidate_id) => {
                if let Some(binding) = self.bindings.get(candidate_id) {
                    self.occurs(binding, variable)
                } else {
                    *candidate_id == variable
                }
            },
            Type::Function(function_type) => {
                function_type.parameters.iter().any(|param| self.occurs(&param.typ, variable))
                    || self.occurs(&function_type.environment, variable)
                    || self.occurs(&function_type.return_type, variable)
                    || self.occurs(&function_type.effects, variable)
            },
            Type::Application(constructor, args) => {
                self.occurs(constructor, variable) || args.iter().any(|arg| self.occurs(arg, variable))
            },
            Type::Forall(_, typ) => self.occurs(typ, variable),
        }
    }

    /// Retrieve a Type then follow all its type variable bindings so that we only return
    /// `Type::Variable` if the type variable is unbound. Note that this may still return
    /// a composite type such as `Type::Application` with bound type variables within.
    fn follow_type<'a>(&'a self, typ: &'a Type) -> &'a Type {
        typ.follow(&self.bindings)
    }

    fn from_cst_type(&mut self, typ: &cst::Type) -> Type {
        Type::from_cst_type(typ, self.current_resolve(), self.compiler, &mut self.next_type_variable_id)
    }

    /// Try to retrieve the types of each field of the given type.
    /// Returns an empty map if unsuccessful.
    ///
    /// The map maps from the field name to a pair of (field type, field index).
    fn get_field_types(&mut self, typ: &Type, generic_args: Option<&[Type]>) -> BTreeMap<Arc<String>, (Type, u32)> {
        match self.follow_type(typ) {
            Type::Application(constructor, arguments) => {
                // TODO: Error if `generic_args` is non-empty
                let constructor = constructor.clone();
                let arguments = arguments.clone();
                // If the constructor is a reference kind (mut, ref, imm, uniq), look up the
                // fields of the inner type and wrap each field type in the same reference.
                if matches!(self.follow_type(&constructor), Type::Primitive(PrimitiveType::Reference(_))) {
                    let inner = arguments[0].clone();
                    let inner_fields = self.get_field_types(&inner, None);
                    return inner_fields
                        .into_iter()
                        .map(|(name, (field_type, index))| {
                            let ref_field = Type::Application(constructor.clone(), Arc::new(vec![field_type]));
                            (name, (ref_field, index))
                        })
                        .collect();
                }
                self.get_field_types(&constructor, Some(&arguments))
            },
            Type::UserDefined(origin) => {
                if let Origin::TopLevelDefinition(id) = origin {
                    let body = id.top_level_item.type_body(generic_args, self.compiler);
                    if let TypeBody::Product { fields, .. } = body {
                        let fields = fields.into_iter().enumerate();
                        return fields.map(|(i, (name, typ))| (name, (typ, i as u32))).collect();
                    }
                }
                BTreeMap::default()
            },
            Type::Primitive(types::PrimitiveType::String) => {
                let mut fields = BTreeMap::default();

                let c_string_type = Type::Application(Arc::new(Type::POINTER), Arc::new(vec![Type::CHAR]));

                // TODO: Hide these and only expose them as unsafe builtins
                fields.insert(Arc::new("c_string".into()), (c_string_type, 0));
                fields.insert(Arc::new("length".into()), (Type::U32, 1));
                fields
            },
            _ => BTreeMap::default(),
        }
    }

    /// Returns a set of substitutions for a user-defined type to replace instances of its generics
    /// with the given types. Care should be taken with the resulting substitutions map since the
    /// Generics within will each be `Origin::Local(name_id)` with a `name_id` local to the given
    /// TypeDefinition, which is likely in a different context than the rest of the TypeChecker.
    ///
    /// Typically, these substitutions can be used on a type within the given TypeDefinition via
    /// a combination of `convert_foreign_type` and `substitute_generics`.
    ///
    /// Does nothing if `replacements.len() != definition.generics.len()`
    fn datatype_generic_substitutions(
        definition: &cst::TypeDefinition, replacements: &[Type],
    ) -> FxHashMap<Generic, Type> {
        let mut substitutions = FxHashMap::default();
        if definition.generics.len() == replacements.len() {
            for (generic, replacement) in definition.generics.iter().zip(replacements) {
                substitutions.insert(Generic::Named(Origin::Local(*generic)), replacement.clone());
            }
        }
        substitutions
    }
}

#[derive(Debug)]
pub enum TypeBody {
    Product { type_name: Name, fields: Vec<(Name, Type)> },
    Sum(Vec<(Name, Vec<Type>)>),
}

impl TopLevelId {
    /// Returns the body of this user-defined type (the part after the `=` when declared).
    /// The given [TopLevelId] should refer to a [TypeDefinition] or something which desugars to
    /// one.
    ///
    /// If specified, `arguments` will be used to substitute any generics of the type.
    /// Panics if the arguments are specified and differ in length to the type's generics.
    ///
    /// Note that if `arguments` are not provided, the type will be instantiated and thus
    /// any fields may refer to type type variables that have not been tracked.
    ///
    /// - For a struct: returns each field name & type
    /// - For a union: returns each variant with its name and arguments
    ///
    /// TODO: This function is called somewhat often but is a lot of work to redo each time.
    pub fn type_body<Db>(self, arguments: Option<&[Type]>, compiler: &Db) -> TypeBody
    where
        Db: DbGet<TypeCheck> + DbGet<GetItem>,
    {
        let result = TypeCheck(self).get(compiler);
        let (item, item_context) = GetItem(self).get(compiler);

        let TopLevelItemKind::TypeDefinition(type_definition) = &item.kind else {
            panic!("type_body: passed type_id is not a type!")
        };

        match &type_definition.body {
            cst::TypeDefinitionBody::Struct(fields) => {
                // This'd be easier with an explicit type data field
                let constructor_type = result.get_generalized(type_definition.name);
                let constructor = apply_type_constructor(constructor_type, arguments, &result);
                let field_types = constructor.function_parameter_types();

                assert_eq!(fields.len(), field_types.len());
                let fields = mapvec(fields.iter().zip(field_types), |((field_name, _), typ)| {
                    (item_context.names[*field_name].clone(), typ)
                });

                let type_name = item_context.names[type_definition.name].clone();
                TypeBody::Product { type_name, fields }
            },
            cst::TypeDefinitionBody::Enum(variants) => {
                let variants = mapvec(variants, |(name, _)| {
                    let constructor_type = result.get_generalized(*name);
                    let constructor = apply_type_constructor(constructor_type, arguments, &result);
                    let fields = constructor.function_parameter_types().collect();
                    (item_context.names[*name].clone(), fields)
                });
                TypeBody::Sum(variants)
            },
            // TODO: Type aliases
            cst::TypeDefinitionBody::Alias(_) | cst::TypeDefinitionBody::Error => {
                // Just make some filler value - ideally we should return an error flag here
                // to prevent future errors
                let type_name = item_context.names[type_definition.name].clone();
                TypeBody::Product { type_name, fields: Vec::new() }
            },
        }
    }
}

/// Try to apply the given type to the given type arguments. Note that this assumes there are no
/// bound type variables within `typ`!
///
// This assumes constructor args are in the same order as the type args.
// This should be guaranteed by [TypeChecker::build_constructor_type].
fn apply_type_constructor(typ: &Type, args: Option<&[Type]>, types: &TypeCheckResult) -> Type {
    let expected_generic_count = match typ.follow(&types.bindings) {
        Type::Forall(generics, _) => generics.len(),
        _ => 0,
    };

    let arg_len = args.map_or(0, |args| args.len());
    if arg_len != expected_generic_count {
        // TODO: We should be issuing an error either here or above somewhere
    }

    let no_type_var_bindings = TypeBindings::default();

    match args {
        Some(args) => {
            if args.len() < expected_generic_count {
                let mut new_args = args.to_vec();
                for _ in args.len()..expected_generic_count {
                    new_args.push(Type::ERROR);
                }
                typ.apply_type(&new_args, &no_type_var_bindings)
            } else {
                typ.apply_type(args, &no_type_var_bindings)
            }
        },
        None if expected_generic_count == 0 => typ.clone(),
        None => {
            // TODO: This should be an error in the future
            let Type::Forall(generics, _) = typ.follow(&types.bindings) else { unreachable!() };
            let args = mapvec(generics.iter(), |_| Type::ERROR);
            typ.apply_type(&args, &no_type_var_bindings)
        },
    }
    .follow_all(&types.bindings)
}

/// Returns each argument of the given function type.
/// If the given type is not a function, an empty Vec is returned.
impl Type {
    fn function_parameter_types(&self) -> impl ExactSizeIterator<Item = Type> {
        let parameters = match self {
            Type::Function(function) => function.parameters.as_slice(),
            _ => &[],
        };
        parameters.iter().map(|param| param.typ.clone())
    }
}
