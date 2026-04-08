use std::{borrow::Cow, collections::BTreeMap, sync::Arc};

use crate::{
    diagnostics::{Diagnostic, UnimplementedItem},
    incremental::{ExportedDefinitions, GetItemRaw, GetType, Resolve},
    iterator_extensions::mapvec,
    name_resolution::{Origin, builtin::Builtin},
    parser::{
        cst::{self, Definition, Expr, Literal, Pattern},
        ids::{ExprId, NameId, PathId, PatternId, TopLevelName},
    },
    type_inference::{
        Locateable, TypeChecker,
        errors::TypeErrorKind,
        get_type::try_get_partial_type,
        types::{self, Type},
    },
};

impl<'local, 'inner> TypeChecker<'local, 'inner> {
    pub(super) fn check_definition(&mut self, definition: &Definition) {
        let expected_type = try_get_partial_type(
            definition, self.current_context(), &self.current_resolve(), self.compiler,
            &mut self.next_type_variable_id,
        );
        let expected_type = match expected_type {
            // Ignore a possible `forall` here, we don't support polymorphic recursion
            Some(typ) => typ.ignore_forall().clone(),
            None => self.next_type_variable(),
        };

        self.check_pattern(definition.pattern, &expected_type);

        // If the RHS is a lambda, call check_lambda directly so we can pass the definition's
        // own name as `self_name`. This prevents self-recursive local functions (such as the
        // `recur` helper produced by loop desugaring) from treating themselves as a captured
        // free variable.
        let self_name = match &self.current_extended_context()[definition.pattern] {
            Pattern::Variable(name) => Some(*name),
            _ => None,
        };

        let rhs = definition.rhs;
        let rhs_expr = match self.current_extended_context().extended_expr(rhs) {
            Some(e) => Cow::Owned(e.clone()),
            None => Cow::Borrowed(&self.current_context()[rhs]),
        };

        match rhs_expr.as_ref() {
            Expr::Lambda(lambda) => {
                let lambda = lambda.clone();
                self.expr_types.insert(rhs, expected_type.clone());
                self.check_lambda(&lambda, &expected_type, rhs, self_name);
            },
            _ => self.check_expr(rhs, &expected_type),
        }

        if definition.implicit {
            self.add_implicit(definition.pattern);
        }
    }

    /// Check an expression's type matches the expected type.
    fn check_expr(&mut self, id: ExprId, expected: &Type) {
        self.expr_types.insert(id, expected.clone());

        let expr = match self.current_extended_context().extended_expr(id) {
            Some(expr) => Cow::Owned(expr.clone()),
            None => Cow::Borrowed(&self.current_context()[id]),
        };

        match expr.as_ref() {
            Expr::Literal(literal) => self.check_literal(literal, id, expected),
            Expr::Variable(path) => self.check_path(*path, expected, Some(id)),
            Expr::Call(call) => self.check_call(call, expected, id),
            Expr::Lambda(lambda) => self.check_lambda(lambda, expected, id, None),
            Expr::Sequence(items) => {
                self.push_implicits_scope();
                for (i, item) in items.iter().enumerate() {
                    let expected_type = if i == items.len() - 1 { expected } else { &self.next_type_variable() };
                    self.check_expr(item.expr, expected_type);
                }
                self.pop_implicits_scope();
            },
            Expr::Definition(definition) => {
                self.check_definition(definition);
                self.unify(&Type::UNIT, expected, TypeErrorKind::General, id);
            },
            Expr::MemberAccess(member_access) => self.check_member_access(member_access, expected, id),
            Expr::If(if_) => self.check_if(if_, expected, id),
            Expr::Match(match_) => self.check_match(match_, expected, id),
            Expr::Reference(reference) => self.check_reference(reference, expected, id),
            Expr::TypeAnnotation(type_annotation) => {
                let annotation = self.from_cst_type(&type_annotation.rhs);
                self.unify(expected, &annotation, TypeErrorKind::TypeAnnotationMismatch, id);
                self.check_expr(type_annotation.lhs, &annotation);
            },
            Expr::Handle(handle) => self.check_handle(handle, expected, id),
            Expr::Constructor(constructor) => self.check_constructor(constructor, expected, id),
            Expr::Quoted(_) => {
                let location = id.locate(self);
                UnimplementedItem::Comptime.issue(self.compiler, location);
            },
            Expr::Loop(_) => unreachable!("Loops should be desugared before type inference"),
            Expr::Return(return_) => self.check_return(return_.expression, id),
            Expr::Assignment(assignment) => self.check_assignment(assignment, expected, id),
            Expr::Error => (),
            Expr::Extern(_) => (),
        }
    }

    fn check_literal(&mut self, literal: &Literal, locator: impl Locateable + Copy, expected: &Type) {
        let actual = match literal {
            Literal::Unit => Type::UNIT,
            Literal::Integer(value, Some(kind)) => {
                self.check_int_fits(*value, *kind, locator);
                Type::integer(*kind)
            },
            Literal::Float(_, Some(kind)) => Type::float(*kind),
            Literal::Bool(_) => Type::BOOL,
            Literal::Integer(value, None) => {
                let type_variable = self.next_type_variable_id();
                self.push_inferred_int(*value, type_variable, locator.locate(self));
                Type::Variable(type_variable)
            },
            Literal::Float(_, None) => {
                let type_variable = self.next_type_variable_id();
                self.push_inferred_float(type_variable, locator.locate(self));
                Type::Variable(type_variable)
            }
            Literal::String(_) => self.get_string_type(),
            Literal::Char(_) => Type::CHAR,
        };
        self.unify(&actual, expected, TypeErrorKind::General, locator);
    }

    pub(super) fn check_name(&mut self, name: NameId, actual: &Type) {
        if let Some(existing) = self.name_types.get(&name) {
            self.unify(actual, &existing.clone(), TypeErrorKind::General, name);
        } else {
            self.name_types.insert(name, actual.clone());
        }
    }

    fn check_pattern(&mut self, id: PatternId, expected: &Type) {
        self.pattern_types.insert(id, expected.clone());

        let pattern = match self.current_extended_context().extended_pattern(id) {
            Some(pattern) => Cow::Owned(pattern.clone()),
            None => Cow::Borrowed(&self.current_context()[id]),
        };

        match pattern.as_ref() {
            Pattern::Error => (),
            Pattern::Variable(name) | Pattern::MethodName { item_name: name, .. } => {
                self.current_lambda_locals.insert(*name);
                self.check_name(*name, expected);
            },
            Pattern::Literal(literal) => self.check_literal(literal, id, expected),
            Pattern::Constructor(path, args) => {
                let parameters = mapvec(args, |_| types::ParameterType::explicit(self.next_type_variable()));

                let expected_function_type = if args.is_empty() {
                    expected.clone()
                } else {
                    Type::Function(Arc::new(types::FunctionType {
                        parameters: parameters.clone(),
                        // Any type constructor we can match on shouldn't be a closure
                        environment: Type::NO_CLOSURE_ENV,
                        return_type: expected.clone(),
                        effects: self.next_type_variable(),
                    }))
                };

                self.check_path(*path, &expected_function_type, None);
                for (expected_arg_type, arg) in parameters.into_iter().zip(args) {
                    self.check_pattern(*arg, &expected_arg_type.typ);
                }
            },
            Pattern::TypeAnnotation(inner_pattern, typ) => {
                let annotated = self.from_cst_type(typ);
                self.unify(expected, &annotated, TypeErrorKind::TypeAnnotationMismatch, id);
                self.check_pattern(*inner_pattern, expected);
            },
        };
    }

    fn check_path(&mut self, path: PathId, expected: &Type, expr: Option<ExprId>) {
        let actual = match self.path_origin(path) {
            Some(Origin::TopLevelDefinition(id)) => self.type_of_top_level_name(&id, path),
            Some(Origin::Local(name)) => self.name_types[&name].clone(),
            Some(Origin::TypeResolution) => self.resolve_type_resolution(path, expected),
            Some(Origin::Builtin(builtin)) => self.check_builtin(builtin, path),
            None => return,
        };
        if let Some(expr) = expr {
            if self.try_coercion(&actual, expected, expr) {
                self.check_expr(expr, expected);
                return;
                // no need to unify or modify self.path_types, that will be handled in the
                // recursive check_expr call since we've just changed the expression at this ExprId.
            }
        }
        self.unify(&actual, expected, TypeErrorKind::General, path);
        self.path_types.insert(path, actual);
    }

    /// Returns the instantiated type of the given TopLevelName.
    ///
    /// Stores the result of the instantiation (if any) to the given [PathId].
    pub(super) fn type_of_top_level_name(&mut self, name: &TopLevelName, path: PathId) -> Type {
        if let Some(typ) = self.item_types.get(name) {
            typ.clone()
        } else {
            let typ = GetType(*name).get(self.compiler);
            let (typ, bindings) = self.instantiate(typ);
            if let Some(bindings) = bindings {
                self.current_extended_context_mut().insert_instantiation(path, bindings);
            }
            typ
        }
    }

    /// Returns the type of a [TopLevelName], possibly instantiating it and returning the bindings,
    /// if any, along with the type.
    pub(super) fn type_and_bindings_of_top_level_name(&mut self, name: &TopLevelName) -> (Type, Option<Vec<Type>>) {
        if let Some(typ) = self.item_types.get(name) {
            (typ.clone(), None)
        } else {
            let typ = GetType(*name).get(self.compiler);
            self.instantiate(typ)
        }
    }

    /// Instantiate the given type, returning the instantiated type and the instantiation bindings.
    ///
    /// This function should not be used outside of [Self::type_and_bindings_of_top_level_name] or [Self::type_of_top_level_name]
    /// since the resulting bindings always need to be remembered.
    fn instantiate(&mut self, typ: Type) -> (Type, Option<Vec<Type>>) {
        match typ {
            Type::Forall(generics, old_type) => {
                assert!(!generics.is_empty());
                let substitutions = generics.iter().map(|generic| (*generic, self.next_type_variable())).collect();
                let typ = old_type.substitute(&substitutions, &self.bindings);

                let bindings = mapvec(generics.iter(), |generic| substitutions[generic].clone());
                (typ, Some(bindings))
            },
            other => (other, None),
        }
    }

    fn resolve_type_resolution(&mut self, path: PathId, expected: &Type) -> Type {
        let path_value = &self.current_context()[path];
        assert_eq!(path_value.components.len(), 1, "Only single-component paths should have Origin::TypeResolution");
        let name = path_value.last_ident();

        let Some(id) = self.try_find_type_namespace_for_type_resolution(expected, name) else {
            return self.issue_name_not_in_scope_error(path);
        };

        // Remember what this `Origin::TypeResolution` path actually refers to from now on
        self.current_extended_context_mut().insert_path_origin(path, Origin::TopLevelDefinition(id));
        self.type_of_top_level_name(&id, path)
    }

    /// Issue a NameNotInScope error and return Type::Error
    fn issue_name_not_in_scope_error(&self, path: PathId) -> Type {
        let name = Arc::new(self.current_context()[path].last_ident().to_owned());
        let location = self.current_context().path_location(path).clone();
        self.compiler.accumulate(Diagnostic::NameNotInScope { name, location });
        Type::ERROR
    }

    fn try_find_type_namespace_for_type_resolution(&self, typ: &Type, constructor_name: &str) -> Option<TopLevelName> {
        match self.follow_type(typ) {
            Type::UserDefined(Origin::TopLevelDefinition(id)) => {
                // We found which type this name belongs to, but if it is a variant we have to
                // check which constructor we want.
                let (_, item_context) = GetItemRaw(id.top_level_item).get(self.compiler);
                let resolve = Resolve(id.top_level_item).get(self.compiler);
                let name_id = resolve
                    .top_level_names
                    .iter()
                    .find(|&&name| item_context.names[name].as_str() == constructor_name)?;

                Some(TopLevelName { top_level_item: id.top_level_item, local_name_id: *name_id })
            },
            Type::Function(function_type) => {
                self.try_find_type_namespace_for_type_resolution(&function_type.return_type, constructor_name)
            },
            Type::Application(constructor, _) => {
                self.try_find_type_namespace_for_type_resolution(constructor, constructor_name)
            },
            _ => None,
        }
    }

    /// Returns the instantiated type of a builtin value
    ///
    /// Will error if passed a builtin type
    fn check_builtin(&mut self, builtin: Builtin, locator: impl Locateable) -> Type {
        match builtin {
            Builtin::Unit => Type::UNIT,
            Builtin::Char | Builtin::Bool | Builtin::Ptr => {
                let typ = Arc::new(builtin.to_string());
                let location = locator.locate(self);
                self.compiler.accumulate(Diagnostic::ValueExpected { location, typ });
                Type::ERROR
            },
            // This needs to match various different function types generally in the form
            // `fn String ... -> a`. For simplicity a type variable is issued here, those working
            // on the stdlib should take care to only use intrinsics with the proper types.
            Builtin::Intrinsic => self.next_type_variable(),
        }
    }

    fn check_call(&mut self, call: &cst::Call, expected: &Type, call_expr: ExprId) {
        // If the function is a MemberAccess, try to resolve it as a method call.
        // `v.push 3` is rewritten to `push (mut v) 3` in the extended context.
        if self.try_rewrite_method_call(call, expected, call_expr) {
            return;
        }

        let expected_parameter_types =
            mapvec(&call.arguments, |arg| types::ParameterType::new(self.next_type_variable(), arg.is_implicit));

        let expected_function_type = {
            let parameters = expected_parameter_types.clone();
            let environment = self.next_type_variable();
            let effects = self.next_type_variable();
            let return_type = expected.clone();
            Type::Function(Arc::new(types::FunctionType { parameters, environment, return_type, effects }))
        };

        self.check_expr(call.function, &expected_function_type);
        for (arg, expected_arg_type) in call.arguments.iter().zip(expected_parameter_types) {
            self.check_expr(arg.expr, &expected_arg_type.typ);
        }
    }

    /// If `call` is `v.push 3` (MemberAccess + args), try to resolve `push` as a function
    /// in the module where `v`'s type is defined. If found, rewrite the Call expression to
    /// `push (mut v) 3` in the extended context and type-check that instead.
    fn try_rewrite_method_call(&mut self, call: &cst::Call, expected: &Type, call_expr: ExprId) -> bool {
        let func_expr = match self.current_extended_context().extended_expr(call.function) {
            Some(expr) => expr.clone(),
            None => self.current_context()[call.function].clone(),
        };

        let Expr::MemberAccess(member_access) = &func_expr else {
            return false;
        };

        let object = member_access.object;
        let member = member_access.member.clone();

        // Type-check the object to learn its type
        let struct_type = self.next_type_variable();
        self.check_expr(object, &struct_type);

        // Find the source file of the type definition
        let Some(source_file) = self.find_type_source_file(&struct_type) else {
            return false;
        };

        let exported = ExportedDefinitions(source_file).get(self.compiler);
        let member_name = Arc::new(member.clone());
        let Some(name) = exported.definitions.get(&member_name) else {
            return false;
        };
        let name = *name;

        let (method_type, bindings) = self.type_and_bindings_of_top_level_name(&name);

        let Type::Function(func_type) = &method_type else {
            return false;
        };

        if func_type.parameters.is_empty() {
            return false;
        }

        let first_param = &func_type.parameters[0].typ;
        let location = self.current_context().expr_location(call.function).clone();

        // Build the object argument, auto-ref'ing if the first parameter is a reference type.
        let object_arg = if let Type::Application(constructor, args) = self.follow_type(first_param) {
            if matches!(self.follow_type(constructor), Type::Primitive(types::PrimitiveType::Reference(..))) {
                let ref_kind = match self.follow_type(constructor) {
                    Type::Primitive(types::PrimitiveType::Reference(k)) => *k,
                    _ => unreachable!(),
                };
                let inner_type = args[0].clone();
                if self.try_unify(&inner_type, &struct_type).is_err() {
                    return false;
                }
                self.unify(&inner_type, &struct_type, TypeErrorKind::General, call_expr);

                let ref_expr = Expr::Reference(cst::Reference { kind: ref_kind, rhs: object });
                self.push_expr(ref_expr, first_param.clone(), location.clone())
            } else {
                if self.try_unify(first_param, &struct_type).is_err() {
                    return false;
                }
                self.unify(first_param, &struct_type, TypeErrorKind::General, call_expr);
                object
            }
        } else {
            if self.try_unify(first_param, &struct_type).is_err() {
                return false;
            }
            self.unify(first_param, &struct_type, TypeErrorKind::General, call_expr);
            object
        };

        // Create a Variable expression for the method, with a fresh path
        let method_path = self.push_path(
            cst::Path { components: vec![(member, location.clone())] },
            method_type.clone(),
            location.clone(),
        );
        self.path_types.insert(method_path, method_type.clone());

        let origin = Origin::TopLevelDefinition(name);
        self.current_extended_context_mut().insert_path_origin(method_path, origin);

        if let Some(bindings) = bindings {
            self.current_extended_context_mut().insert_instantiation(method_path, bindings);
        }

        let method_var = self.push_expr(Expr::Variable(method_path), method_type.clone(), location);

        // Build the new Call: `push (mut v) 3`
        let mut new_arguments = vec![cst::Argument::explicit(object_arg)];
        new_arguments.extend_from_slice(&call.arguments);

        let new_call = Expr::Call(cst::Call { function: method_var, arguments: new_arguments });
        self.current_extended_context_mut().insert_expr(call_expr, new_call);

        // Now type-check the rewritten expression
        self.check_expr(call_expr, expected);
        true
    }

    fn check_lambda(&mut self, lambda: &cst::Lambda, expected: &Type, expr: ExprId, self_name: Option<NameId>) {
        let function_type = match self.follow_type(expected) {
            Type::Function(function_type) => function_type.clone(),
            _ => {
                let parameters = mapvec(&lambda.parameters, |param| {
                    types::ParameterType::new(self.next_type_variable(), param.is_implicit)
                });
                let expected_parameter_count = parameters.len();
                let environment = self.next_type_variable();
                let return_type = self.next_type_variable();
                let effects = self.next_type_variable();
                let new_type = Arc::new(types::FunctionType { parameters, environment, return_type, effects });
                let function_type = Type::Function(new_type.clone());
                self.unify(expected, &function_type, TypeErrorKind::Lambda { expected_parameter_count }, expr);
                new_type
            },
        };

        // Remember the return type so that it can be checked by `return` statements
        let old_return_type =
            std::mem::replace(&mut self.function_return_type, Some(function_type.return_type.clone()));
        let old_lambda_locals = std::mem::take(&mut self.current_lambda_locals);
        if let Some(name) = self_name {
            self.current_lambda_locals.insert(name);
        }

        self.push_implicits_scope();
        self.check_function_parameter_count(&function_type.parameters, lambda.parameters.len(), expr);
        let parameter_lengths_match = function_type.parameters.len() == lambda.parameters.len();

        for (parameter, expected_type) in lambda.parameters.iter().zip(function_type.parameters.iter()) {
            // Avoid extra errors if the parameter length isn't as expected
            let expected_type = if parameter_lengths_match { &expected_type.typ } else { &Type::ERROR };
            self.check_pattern(parameter.pattern, expected_type);

            if parameter.is_implicit {
                self.add_implicit(parameter.pattern);
            }
        }

        // Required in case `function_type` has fewer parameters, to ensure we check all of `lambda.parameters`
        for parameter in lambda.parameters.iter().skip(function_type.parameters.len()) {
            self.check_pattern(parameter.pattern, &Type::ERROR);
        }

        // TODO: Check lambda.effects
        let return_type = if let Some(return_type) = lambda.return_type.as_ref() {
            let return_type = self.from_cst_type(return_type);
            self.unify(&return_type, &function_type.return_type, TypeErrorKind::TypeAnnotationMismatch, expr);
            Cow::Owned(return_type)
        } else {
            Cow::Borrowed(&function_type.return_type)
        };

        self.check_expr(lambda.body, &return_type);

        self.function_return_type = old_return_type;
        self.current_lambda_locals = old_lambda_locals;
        self.pop_implicits_scope();

        // pop_implicits_scope modifies the function by inserting implicit arguments, we need
        // to check captures only after that step in case any of those arguments are captured.
        if self.coercion_wrapper_exprs.contains(&expr) {
            // For coercion wrapper lambdas, the implicit argument slots are filled in by the
            // *enclosing* lambda's pop_implicits_scope. Defer the closure check to that scope
            // so free-variable analysis sees the resolved arguments rather than Expr::Error.
            if let Some(scope) = self.implicits.last_mut() {
                scope.push_deferred_closure_check(expr, function_type.environment.clone());
            }
        } else {
            self.check_for_closure(expr, &function_type.environment, self_name);
        }
    }

    /// Check a function's parameter count using the given parameter types as the expected count.
    /// Issues an error if the expected count does not match the actual count.
    fn check_function_parameter_count(
        &mut self, parameters: &Vec<types::ParameterType>, actual_count: usize, expr: ExprId,
    ) {
        if actual_count != parameters.len() {
            self.compiler.accumulate(Diagnostic::FunctionArgCountMismatch {
                actual: actual_count,
                expected: parameters.len(),
                location: self.current_context().expr_location(expr).clone(),
            });
        }
    }

    fn check_member_access(&mut self, member_access: &cst::MemberAccess, expected: &Type, expr: ExprId) {
        let struct_type = self.next_type_variable();
        self.check_expr(member_access.object, &struct_type);

        let fields = self.get_field_types(&struct_type, None);
        if let Some((field, field_index)) = fields.get(&member_access.member) {
            self.current_extended_context_mut().push_member_access_index(expr, *field_index);

            if self.try_coercion(field, expected, expr) {
                self.check_expr(expr, expected);
            } else {
                self.unify(field, expected, TypeErrorKind::General, expr);
            }
        } else if matches!(self.follow_type(&struct_type), Type::Variable(_)) {
            let location = self.current_context().expr_location(expr).clone();
            self.compiler.accumulate(Diagnostic::TypeMustBeKnownMemberAccess { location });
        } else {
            let typ = self.type_to_string(&struct_type);
            let location = self.current_context().expr_location(expr).clone();
            let name = Arc::new(member_access.member.clone());
            self.compiler.accumulate(Diagnostic::NoSuchFieldForType { typ, location, name });
        }
    }

    /// Find the source file where a type is defined, unwrapping references and type applications.
    fn find_type_source_file(&self, typ: &Type) -> Option<crate::name_resolution::namespace::SourceFileId> {
        match self.follow_type(typ) {
            Type::UserDefined(Origin::TopLevelDefinition(id)) => Some(id.top_level_item.source_file),
            Type::Application(constructor, args) => {
                if matches!(
                    self.follow_type(constructor),
                    Type::Primitive(types::PrimitiveType::Reference(_))
                ) {
                    self.find_type_source_file(&args[0])
                } else {
                    self.find_type_source_file(constructor)
                }
            },
            _ => None,
        }
    }

    fn check_if(&mut self, if_: &cst::If, expected: &Type, expr: ExprId) {
        self.check_expr(if_.condition, &Type::BOOL);

        // If there's an else clause our expected return type should match the then/else clauses'
        // types. Otherwise, the then body may be any type.
        let expected = if if_.else_.is_some() {
            Cow::Borrowed(expected)
        } else {
            self.unify(&Type::UNIT, expected, TypeErrorKind::IfStatement, expr);
            Cow::Owned(self.next_type_variable())
        };

        self.push_implicits_scope();
        self.check_expr(if_.then, &expected);
        self.pop_implicits_scope();

        // TODO: No way to identify if `then_type != else_type`. This would be useful to point out
        // for error messages.
        if let Some(else_) = if_.else_ {
            self.push_implicits_scope();
            self.check_expr(else_, &expected);
            self.pop_implicits_scope();
        }
    }

    fn check_match(&mut self, match_: &cst::Match, expected: &Type, expr: ExprId) {
        let expr_type = self.next_type_variable();

        // Push an implicits scope here so we can default any integers used in the match
        // to an `I32` before the decision tree checks occur. This lets us compile `match 1 | ...`
        // without errors that the type of `1` is not yet known.
        self.push_implicits_scope();
        self.check_expr(match_.expression, &expr_type);

        for (pattern, branch) in match_.cases.iter() {
            self.check_pattern(*pattern, &expr_type);
            // TODO: Specify if branch_type != type of first branch for better error messages
            self.push_implicits_scope();
            self.check_expr(*branch, expected);
            self.pop_implicits_scope();
        }
        self.pop_implicits_scope();

        // Now compile the match into a decision tree. The `match expr | ...` expression will be
        // replaced with `<fresh> = expr; <decision tree>`
        let location = self.current_context().expr_location(match_.expression).clone();
        let (match_var, match_var_name) = self.fresh_variable("match_var", expr_type.clone(), location.clone());

        // `<match_var> = <expression being matched>`
        let preamble = self.let_binding(match_var_name, match_.expression);

        if let Some(tree) = self.compile_decision_tree(match_var, &match_.cases, expr_type, location) {
            let context = self.current_extended_context_mut();
            context.insert_decision_tree(expr, preamble, tree);
        }
    }

    fn check_reference(&mut self, reference: &cst::Reference, expected: &Type, expr: ExprId) {
        let actual = Type::reference(reference.kind);

        let expected_element_type = match self.follow_type(expected) {
            Type::Application(constructor, args) => {
                let constructor = constructor.clone();
                let args = args.clone();
                let first_arg = args.first();

                match self.follow_type(&constructor) {
                    Type::Primitive(types::PrimitiveType::Reference(..)) => {
                        self.unify(&actual, &constructor, TypeErrorKind::ReferenceKind, expr);

                        // Expect incorrect arg counts to be resolved beforehand
                        first_arg.unwrap().clone()
                    },
                    _ => {
                        if self.unify(&actual, expected, TypeErrorKind::ExpectedNonReference, expr) {
                            first_arg.unwrap().clone()
                        } else {
                            Type::ERROR
                        }
                    },
                }
            },
            Type::Variable(id) => {
                let id = *id;
                let element = self.next_type_variable();
                let expected = Type::Application(Arc::new(actual), Arc::new(vec![element.clone()]));
                self.bindings.insert(id, expected);
                element
            },
            _ => {
                self.unify(&actual, expected, TypeErrorKind::ExpectedNonReference, expr);
                Type::ERROR
            },
        };

        self.check_expr(reference.rhs, &expected_element_type);
    }

    fn check_constructor(&mut self, constructor: &cst::Constructor, expected: &Type, id: ExprId) {
        let typ = self.from_cst_type(&constructor.typ);
        self.unify(&typ, expected, TypeErrorKind::Constructor, id);

        // Map each field name to its index in the type's declaration order.
        // This is used when lowering to MIR when structs are converted into tuples.
        let mut field_order = BTreeMap::new();
        let field_types = self.get_field_types(&typ, None);

        for (name, expr) in &constructor.fields {
            let name_string = &self.current_context()[*name];
            let (expected_field_type, field_index) = field_types.get(name_string).cloned().unwrap_or((Type::ERROR, 0));

            self.check_expr(*expr, &expected_field_type);
            self.check_name(*name, &expected_field_type);

            field_order.insert(*name, field_index);
        }

        self.current_extended_context_mut().push_constructor_field_order(id, field_order);
    }

    fn check_handle(&mut self, _handle: &cst::Handle, _expected: &Type, expr: ExprId) {
        let location = self.current_context().expr_location(expr).clone();
        UnimplementedItem::Effects.issue(self.compiler, location);
    }

    fn check_assignment(&mut self, assignment: &cst::Assignment, expected: &Type, id: ExprId) {
        let lhs_type = self.next_type_variable();
        self.check_expr(assignment.lhs, &lhs_type);

        // Check if this assignment mutates a captured variable. Captures are by value,
        // so mutating a capture won't affect the original. Allow if the LHS is a reference
        // type since the assignment stores through the reference.
        let lhs_followed = self.follow_type(&lhs_type);
        let lhs_is_ref = matches!(&lhs_followed, Type::Application(c, _)
            if matches!(self.follow_type(c), Type::Primitive(types::PrimitiveType::Reference(_))));

        if !lhs_is_ref {
            if let Some(root_name) = self.find_lvalue_root_variable(assignment.lhs) {
                if !self.current_lambda_locals.contains(&root_name) {
                    let location = id.locate(self);
                    self.compiler.accumulate(Diagnostic::MutatedCapturedVariable { location });
                }
            }
        }

        // If the LHS is a reference type (e.g. `p.x` where `p: mut Point` yields `mut I32`),
        // the RHS should match the inner (pointee) type rather than the reference wrapper.
        let rhs_type = if lhs_is_ref {
            match self.follow_type(&lhs_type) {
                Type::Application(_, args) => args[0].clone(),
                _ => unreachable!(),
            }
        } else {
            lhs_type.clone()
        };
        self.check_expr(assignment.rhs, &rhs_type);
        self.unify(&Type::UNIT, expected, TypeErrorKind::General, id);
    }

    /// Walk through member accesses to find the root local variable of an lvalue expression.
    fn find_lvalue_root_variable(&self, expr: ExprId) -> Option<NameId> {
        let e = match self.current_extended_context().extended_expr(expr) {
            Some(e) => Cow::Owned(e.clone()),
            None => Cow::Borrowed(&self.current_context()[expr]),
        };
        match e.as_ref() {
            Expr::Variable(path) => {
                if let Some(Origin::Local(name)) = self.path_origin(*path) {
                    Some(name)
                } else {
                    None
                }
            },
            Expr::MemberAccess(access) => self.find_lvalue_root_variable(access.object),
            _ => None,
        }
    }

    /// Return can unify with any type locally so we don't need the expected type here
    fn check_return(&mut self, returned_expr: ExprId, id: ExprId) {
        match self.function_return_type.as_ref().cloned() {
            Some(expected_return) => {
                self.check_expr(returned_expr, &expected_return);
            },
            None => {
                let location = id.locate(self);
                self.compiler.accumulate(Diagnostic::ReturnNotInFunction { location });
            },
        }
    }

    pub(super) fn check_comptime(&self, _comptime: &cst::Comptime) {
        let location = self.current_context().location().clone();
        UnimplementedItem::Comptime.issue(self.compiler, location);
    }
}
