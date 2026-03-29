use std::sync::Arc;

use crate::{
    diagnostics::{Diagnostic, Location},
    incremental::VisibleImplicits,
    iterator_extensions::mapvec,
    lexer::token::IntegerKind,
    name_resolution::Origin,
    parser::{
        cst::{self, Name, Pattern},
        ids::{ExprId, PatternId},
    },
    type_inference::{
        DelayedImplicit, Locateable, TypeChecker,
        types::{FunctionType, ParameterType, PrimitiveType, Type, TypeVariableId},
    },
};

/// Any more than this arbitrary value and we stop looking for impls to populate the error
/// message with and instead display `..`.
const MULTIPLE_MATCHING_IMPLS_CUTOFF: usize = 5;

impl<'local, 'inner> TypeChecker<'local, 'inner> {
    /// Perform an implicit parameter coercion.
    ///
    /// Given a function `expr` which requires some implicit parameters present in the `actual`
    /// type but not the `expected` type, find values for those implicits (issuing errors for any
    /// that cannot be found) an create a new wrapper function. E.g:
    ///
    /// ```ante
    /// fn a c -> <expr> a <new-implicit> c
    /// ```
    /// where `<new-implicit>` is a new implicit that was successfully found. In the case a
    /// matching implicit value cannot be found, an error is issued and an error expression is
    /// slotted in as the argument instead. In this way, this function will always return a new
    /// closure wrapper.
    pub(super) fn implicit_parameter_coercion(
        &mut self, actual: Arc<FunctionType>, expected: Arc<FunctionType>, function: ExprId,
    ) -> Option<cst::Expr> {
        // Looking for implicit parameters that are in `actual` but not `expected`.
        // The reverse would be a type error.
        let mut new_expected = Vec::new();

        let mut actual_params = actual.parameters.iter();
        let mut expected_params = expected.parameters.iter().cloned();
        let mut current_expected = expected_params.next();

        // For each parameter, this is either `None` if no new implicit was inserted
        // at that position, or it is `Some(expr_id)` of the new expression.
        let mut implicits_added = Vec::new();

        while let Some(actual) = actual_params.next() {
            match (actual.is_implicit, current_expected.as_ref()) {
                // actual is implicit, but expected isn't, search for an implicit in scope
                (true, expected) if expected.map_or(true, |param| !param.is_implicit) => {
                    let value = self.delay_find_implicit_value(&actual.typ, new_expected.len(), function);
                    implicits_added.push(Some(value));
                    new_expected.push(ParameterType::implicit(self.expr_types[&value].clone()));
                },
                _ => {
                    let expected = current_expected.unwrap_or(ParameterType::explicit(Type::ERROR));
                    new_expected.push(expected);
                    implicits_added.push(None);
                    current_expected = expected_params.next();
                },
            }
        }
        self.create_closure_wrapper_for_implicit(function, implicits_added, new_expected)
    }

    /// If the expression is a variable, return its name
    fn try_get_name(&self, expr: ExprId) -> Option<String> {
        match &self.current_extended_context()[expr] {
            cst::Expr::Variable(path) => Some(self.current_extended_context()[*path].last_ident().to_string()),
            _ => None,
        }
    }

    pub(super) fn push_implicits_scope(&mut self) {
        self.implicits.push(Default::default());
    }

    pub(super) fn pop_implicits_scope(&mut self) {
        // We must perform any queued requests before popping any implicits which should be visible
        if let Some(scope) = self.implicits.last_mut() {
            let implicits = std::mem::take(&mut scope.delayed_implicits);
            let closures = std::mem::take(&mut scope.deferred_closure_checks);
            let integers = std::mem::take(&mut scope.integer_type_variables);

            // Phase 1: Try to resolve all implicits. When the target type contains an unbound
            // integer type variable, unification may still succeed (e.g. searching for `Foo _`
            // where only `Foo U8` exists binds `_ := U8`). Failures are collected for retry.
            let mut failed_implicits = Vec::new();
            for implicit in implicits {
                if let Err(error) = self.find_implicit_value(implicit) {
                    failed_implicits.push((implicit, error));
                }
            }

            // Default any still-unbound integers to I32 and ensure their value fits
            // in whatever type they are now.
            for (value, type_variable, location) in integers {
                self.try_default_integer_to_i32(value, type_variable, location);
            }

            // Phase 2: Retry implicits that failed in phase 1, now that integer type variables
            // have been defaulted to I32. This handles cases like `Add _` where phase 1 finds
            // multiple `Add X` candidates (ambiguous on an unbound integer), but after defaulting
            // `_ := I32` exactly one candidate remains. If the retry still fails, accumulate the
            // original error so the diagnostic reflects the unbound type the user wrote.
            for (implicit, original_error) in failed_implicits {
                if let Err(_) = self.find_implicit_value(implicit) {
                    self.compiler.accumulate(original_error);
                }
            }

            // Run deferred closure checks after all implicits (including retries) are resolved
            // so that free-variable analysis sees the fully-resolved implicit arguments.
            for (expr, env_type) in closures {
                self.check_for_closure(expr, &env_type, None);
            }
        }

        self.implicits.pop().expect("More pops than pushes to `TypeChecker::implicits`");
    }

    pub(super) fn push_inferred_int(&mut self, value: u64, type_variable: TypeVariableId, location: Location) {
        self.implicits.last_mut().unwrap().integer_type_variables.push((value, type_variable, location));
    }

    /// Delay finding an implicit value until later when more types are known.
    ///
    /// This returns a fresh [ExprId] where the implicit value will be emplaced into when found.
    fn delay_find_implicit_value(&mut self, target_type: &Type, parameter_index: usize, function: ExprId) -> ExprId {
        let location = function.locate(self);
        let typ = target_type.clone();
        let fresh_id = self.push_expr(cst::Expr::Error, typ, location);
        let delayed = DelayedImplicit { source: function, destination: fresh_id, parameter_index };
        self.implicits.last_mut().unwrap().delayed_implicits.push(delayed);
        fresh_id
    }

    fn try_default_integer_to_i32(&mut self, value: u64, type_variable: TypeVariableId, location: Location) {
        let kind = match Type::Variable(type_variable).follow(&self.bindings) {
            Type::Variable(id) => {
                self.bindings.insert(*id, Type::Primitive(PrimitiveType::Int(IntegerKind::I32)));
                IntegerKind::I32
            },
            Type::Primitive(PrimitiveType::Int(kind)) => *kind,
            // Bound to a non-integer type. Invalid, but we rely on unification to produce more
            // localized errors for this case.
            _ => return,
        };

        // Now ensure the literal fits in the chosen kind
        self.check_int_fits(value, kind, location);
    }

    /// Search for an implicit value in scope with the given type, returning an error if no implicit
    /// is found or if multiple matching implicits are found.
    ///
    /// If an implicit is found, its value is emplaced into the `destination` expression.
    fn find_implicit_value(&mut self, implicit: DelayedImplicit) -> Result<(), Diagnostic> {
        let target_type = self.expr_types[&implicit.destination].clone();
        let parameter_index = implicit.parameter_index;
        let function = implicit.source;
        let destination = implicit.destination;

        // A Vec of (implicit name, implicit origin, implicit type, implicit arguments)
        // Non-function implicits will not have any arguments
        let mut candidates = Candidates::new();

        // Parallel vec tracking the bindings committed by each candidate's check. We save and
        // restore bindings around each check so that the first candidate's successful try_unify
        // doesn't permanently commit type variables (e.g. binding a free variable to I16 just
        // because add_i16 comes before add_i32 alphabetically in VisibleImplicits). After
        // collecting all candidates we apply only the single winner's bindings.
        let mut candidate_bindings = Vec::new();

        // TODO: Remove clone by making try_unify no longer require a mutable self
        let implicits_in_scope =
            self.implicits.iter().rev().flat_map(|scope| &scope.implicits_in_scope).copied().collect::<Vec<_>>();

        for name in implicits_in_scope {
            if candidates.len() > MULTIPLE_MATCHING_IMPLS_CUTOFF {
                // Multiple matching impls, don't waste time looking for more.
                // There are many Eq impls we could waste time on for example.
                break;
            }

            let saved = self.bindings.clone();
            let name_type = self.name_types[&name].follow(&self.bindings).clone();
            let origin = Origin::Local(name);
            let name = self.current_extended_context()[name].clone();
            let prev_len = candidates.len();
            self.check_implicit_candidate(
                name_type,
                None,
                &target_type,
                name,
                origin,
                &mut candidates,
                parameter_index,
                function,
            );
            let committed = std::mem::replace(&mut self.bindings, saved);
            if candidates.len() > prev_len {
                candidate_bindings.push(committed);
            }
        }

        // Need to check globally visible implicits separately
        // TODO: Make this more efficient so we don't need to go through every single implicit
        if candidates.len() <= MULTIPLE_MATCHING_IMPLS_CUTOFF
            && let Some(item) = self.current_item
        {
            for (name, name_id) in VisibleImplicits(item.source_file).get(self.compiler).iter() {
                if candidates.len() > MULTIPLE_MATCHING_IMPLS_CUTOFF {
                    // Multiple matching impls, don't waste time looking for more.
                    // There are many Eq impls we could waste time on for example.
                    break;
                }

                let (name_type, bindings) = self.type_and_bindings_of_top_level_name(name_id);
                let saved = self.bindings.clone();
                // Pre-filter: skip impls that clearly can't match. After the pre-filter succeeds
                // we restore bindings so the committed state is captured per-candidate below.
                if self.try_unify(&name_type, &target_type).is_ok() {
                    self.bindings = saved.clone();
                    let prev_len = candidates.len();
                    let origin = Origin::TopLevelDefinition(*name_id);
                    self.check_implicit_candidate(
                        name_type,
                        bindings,
                        &target_type,
                        name.clone(),
                        origin,
                        &mut candidates,
                        parameter_index,
                        function,
                    );
                    let committed = std::mem::replace(&mut self.bindings, saved);
                    if candidates.len() > prev_len {
                        candidate_bindings.push(committed);
                    }
                } else {
                    self.bindings = saved;
                }
            }
        }

        if candidates.is_empty() {
            Err(self.no_implicit_found_error(&target_type, parameter_index, function))
        } else if candidates.len() == 1 {
            self.bindings = candidate_bindings.remove(0);
            let (name, origin, bindings, name_type, arguments) = candidates.remove(0);
            let location = function.locate(self);
            self.create_implicit_argument_expr(name, origin, bindings, name_type, arguments, destination, location);
            Ok(())
        } else {
            Err(self.multiple_matching_implicits_error(candidates, &target_type, parameter_index, function))
        }
    }

    /// Check if the given `implicit_type` matches the `target_type` directly, or if it can be
    /// called as a function to produce the target type. If either are true, push the candidate to
    /// the candidates list.
    fn check_implicit_candidate(
        &mut self, implicit_type: Type, type_bindings: Option<Vec<Type>>, target_type: &Type, name: Name,
        origin: Origin, candidates: &mut Candidates, parameter_index: usize, function: ExprId,
    ) {
        match self.implicit_type_matches(&implicit_type, target_type) {
            ImplicitMatch::NoMatch => (),
            ImplicitMatch::MatchedAsIs => {
                candidates.push((name, origin, type_bindings, implicit_type, Vec::new()));
            },
            ImplicitMatch::Call(function_type) => {
                // TODO: Make this algorithm iterative instead of recursive
                let mut arguments = Vec::new();
                for parameter in &function_type.parameters {
                    if parameter.is_implicit {
                        let arg_type = parameter.typ.clone();
                        let arg_location = function.locate(self);
                        let destination = self.push_expr(cst::Expr::Error, arg_type, arg_location);
                        let implicit = DelayedImplicit { source: function, destination, parameter_index };

                        if self.find_implicit_value(implicit).is_ok() {
                            arguments.push(cst::Argument::implicit(destination));
                        }
                    }
                }

                if arguments.len() == function_type.parameters.len() {
                    candidates.push((name, origin, type_bindings, implicit_type, arguments));
                }
            },
        }
    }

    /// Given the type of an implicit value, and the target type to search for, return whether the
    /// given implicit is a match for the target type, whether it can produce such a type by
    /// calling it as a function, or whether there is no match.
    fn implicit_type_matches(&mut self, implicit_type: &Type, target_type: &Type) -> ImplicitMatch {
        if self.try_unify(implicit_type, target_type).is_ok() {
            ImplicitMatch::MatchedAsIs
        } else if let Type::Function(f) = implicit_type {
            if self.try_unify(&f.return_type, target_type).is_ok() {
                ImplicitMatch::Call(f.clone())
            } else {
                ImplicitMatch::NoMatch
            }
        } else {
            ImplicitMatch::NoMatch
        }
    }

    // error: No implicit found for parameter N of type T
    fn no_implicit_found_error(&self, implicit_type: &Type, parameter_index: usize, function: ExprId) -> Diagnostic {
        let type_string = self.type_to_string(&implicit_type);
        let function_name = self.try_get_name(function);
        let location = function.locate(self);
        Diagnostic::NoImplicitFound { type_string, function_name, parameter_index, location }
    }

    // error: No implicit found for parameter N of type T
    fn multiple_matching_implicits_error(
        &self, matching: Candidates, implicit_type: &Type, parameter_index: usize, function: ExprId,
    ) -> Diagnostic {
        let type_string = self.type_to_string(&implicit_type);
        let function_name = self.try_get_name(function);
        let location = function.locate(self);

        let mut matches = mapvec(matching, |(name, _, _, _, _)| name);
        if matches.len() > MULTIPLE_MATCHING_IMPLS_CUTOFF {
            matches.truncate(MULTIPLE_MATCHING_IMPLS_CUTOFF);
            matches.push(Arc::new("..".to_string()));
        }

        Diagnostic::MultipleImplicitsFound { matches, type_string, function_name, parameter_index, location }
    }

    /// Try to add the given implicit into scope
    pub(super) fn add_implicit(&mut self, id: PatternId) {
        let name = match &self.current_extended_context()[id] {
            Pattern::Error => return,
            Pattern::Variable(name) => *name,
            Pattern::TypeAnnotation(inner_id, _) => return self.add_implicit(*inner_id),
            _ => {
                let location = id.locate(self);
                self.compiler.accumulate(Diagnostic::ImplicitNotAVariable { location });
                return;
            },
        };
        self.implicits.last_mut().unwrap().implicits_in_scope.push(name);
    }

    /// Given:
    /// - A function `f`
    /// - `implicits_added = [None, Some(i), None]` (e.g.)
    /// - `argument_types = [t, u, v]`
    ///
    /// Create:
    /// `fn (a: t) (c: v) -> f a {i} c`
    fn create_closure_wrapper_for_implicit(
        &mut self, function: ExprId, implicits_added: Vec<Option<ExprId>>, argument_types: Vec<ParameterType>,
    ) -> Option<cst::Expr> {
        // We should always have at least 1 added implicit parameter
        let implicit_added = implicits_added.iter().any(|param| param.is_some());

        // A type-error is expected when type checking this call
        if !implicit_added || implicits_added.len() != argument_types.len() {
            return None;
        }

        let mut parameters = Vec::new();
        let mut arguments = Vec::new();

        for (implicit, arg_type) in implicits_added.into_iter().zip(argument_types) {
            match implicit {
                // We want new implicit arguments to be in the call but not the lambda parameters
                Some(arg) => {
                    arguments.push(cst::Argument::implicit(arg));
                },
                None => {
                    let location = function.locate(self);
                    let (var_path, var_name) = self.fresh_variable("p", arg_type.typ.clone(), location.clone());

                    let pattern = self.push_pattern(cst::Pattern::Variable(var_name), location.clone());
                    let expr = self.push_expr(cst::Expr::Variable(var_path), arg_type.typ, location);

                    arguments.push(cst::Argument { is_implicit: arg_type.is_implicit, expr });
                    parameters.push(cst::Parameter { is_implicit: arg_type.is_implicit, pattern });
                },
            }
        }

        // Since `function` is the ExprId we'll be replacing, we can't use it directly here. We
        // have to copy it to a new id.
        let location = function.locate(self);
        let expr = self.current_extended_context()[function].clone();

        // This type should be overwritten later when cst_traversal traverses this new expr
        let function = self.push_expr(expr, Type::ERROR, location.clone());

        let body = cst::Expr::Call(cst::Call { function, arguments });
        let body_type = Type::ERROR;
        let body = self.push_expr(body, body_type, location);

        // TODO: This should have the same effects as `function`
        Some(cst::Expr::Lambda(cst::Lambda { parameters, body, return_type: None, effects: Some(Vec::new()) }))
    }

    /// Creates a new expression referring to the given implicit value.
    /// - 0 arguments: The expression is a variable
    /// - 1+ arguments: The expression is a function call to the given name, using the given arguments.
    fn create_implicit_argument_expr(
        &mut self, name: Name, origin: Origin, type_bindings: Option<Vec<Type>>, name_type: Type,
        arguments: Vec<cst::Argument>, destination: ExprId, location: Location,
    ) {
        let name = name.as_ref().clone();
        let path = self.push_path(cst::Path::ident(name, location.clone()), name_type.clone(), location.clone());
        let variable = cst::Expr::Variable(path);

        let context = self.current_extended_context_mut();
        context.insert_path_origin(path, origin);

        if let Some(bindings) = type_bindings {
            context.insert_instantiation(path, bindings);
        }

        let (expr, typ) = if arguments.is_empty() {
            (variable, name_type)
        } else {
            let return_type = name_type.return_type().unwrap().clone();
            let function = self.push_expr(variable, name_type, location.clone());
            let call = cst::Expr::Call(cst::Call { function, arguments });
            (call, return_type)
        };

        self.current_extended_context_mut().insert_expr(destination, expr);
        self.expr_types.insert(destination, typ);
    }
}

/// Candidates when searching for an implicit value.
/// Contains the name, origin, instantiation type bindings, type for the implicit, along with any arguments to
/// call it with (if any) if we should call this implicit for its return value.
type Candidates = Vec<(Name, Origin, Option<Vec<Type>>, Type, Vec<cst::Argument>)>;

enum ImplicitMatch {
    NoMatch,
    MatchedAsIs,
    Call(Arc<FunctionType>),
}
