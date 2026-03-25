use std::sync::Arc;

use crate::{
    diagnostics::Location,
    incremental::{DbHandle, GetItem, GetItemRaw},
    iterator_extensions::mapvec,
    parser::{
        context::TopLevelContext,
        cst::{
            self, Constructor, Definition, Expr, Lambda, Pattern, TopLevelItem, TopLevelItemKind, TraitDefinition,
            TraitImpl, Type, TypeDefinitionBody, TypeKind,
        },
        ids::ExprId,
    },
};

fn make_tuple_type(location: &Location, mut types: impl ExactSizeIterator<Item = Type>) -> Type {
    let Some(first) = types.next() else {
        return Type::new(TypeKind::NoClosureEnv, location.clone());
    };

    if types.len() == 0 {
        first
    } else {
        let rest = make_tuple_type(location, types);
        let pair = Type::new(TypeKind::Pair, location.clone());
        Type::new(TypeKind::Application(Box::new(pair), vec![first, rest]), location.clone())
    }
}

pub fn get_item_impl(context: &GetItem, db: &DbHandle) -> (Arc<TopLevelItem>, Arc<TopLevelContext>) {
    let (item, context) = GetItemRaw(context.0).get(db);

    match &item.kind {
        TopLevelItemKind::TraitDefinition(trait_definition) => {
            let mut new_context = context.as_ref().clone();
            let new_kind = desugar_trait(trait_definition, &mut new_context);
            let new_item = Arc::new(TopLevelItem { comments: item.comments.clone(), kind: new_kind, id: item.id });
            (new_item, Arc::new(new_context))
        },
        TopLevelItemKind::TraitImpl(trait_impl) => {
            // TODO: Reduce cloning costs for context, comments
            let mut new_context = context.as_ref().clone();
            let new_kind = desugar_impl(trait_impl, &mut new_context);
            let new_item = Arc::new(TopLevelItem { comments: item.comments.clone(), kind: new_kind, id: item.id });
            (new_item, Arc::new(new_context))
        },
        TopLevelItemKind::Definition(definition) => {
            let mut new_context = context.as_ref().clone();
            let new_kind = desugar_expression(definition.rhs, &mut new_context);
            let new_item = Arc::new(TopLevelItem { comments: item.comments.clone(), kind: new_kind, id: item.id });
            (new_item, Arc::new(new_context))
        },
        _ => (item, context),
    }
}

/// Expands a trait-typed parameter from e.g. `Print a` → `Print a [env_i]`.
/// This ensures `from_cst_type_no_type_variables` sees a named generic for the env
/// rather than auto-inserting a fresh type variable (which would cause it to return None).
fn add_env_to_trait_type(typ: &Type, env_var: crate::parser::ids::NameId, location: &Location) -> Type {
    let env_type = Type::new(TypeKind::Variable(env_var), location.clone());
    match &typ.kind {
        TypeKind::Named(_) => {
            Type::new(TypeKind::Application(Box::new(typ.clone()), vec![env_type]), typ.location.clone())
        },
        TypeKind::Application(f, args) => {
            let mut new_args = args.clone();
            new_args.push(env_type);
            Type::new(TypeKind::Application(f.clone(), new_args), typ.location.clone())
        },
        _ => typ.clone(),
    }
}

/// Desugars
/// ```ante
/// impl name {Parameter}: Trait TraitArgs with
///     method1 = ...
///     method2 = ...
/// ```
/// Into
/// ```ante
/// implicit name {Parameter}: Trait TraitArgs Parameter = Trait With
///     method1 = ...
///     method2 = ...
/// ```
/// Note that this assumes the returned trait will capture each parameter used.
fn desugar_impl(impl_: &TraitImpl, context: &mut TopLevelContext) -> TopLevelItemKind {
    let variable = context.patterns.push(Pattern::Variable(impl_.name));
    let location = context.name_locations[impl_.name].clone();
    assert_eq!(variable, context.pattern_locations.push(location.clone()));

    let mut trait_type = Type::new(TypeKind::Named(impl_.trait_path), location.clone());

    // Collect existing parameter info before mutating context.
    let param_infos: Vec<(bool, crate::parser::ids::PatternId, Type)> = impl_
        .parameters
        .iter()
        .map(|param| match &context.patterns[param.pattern] {
            Pattern::TypeAnnotation(inner, typ) => (param.is_implicit, *inner, typ.clone()),
            _ => unreachable!("impl parameters are expected to have type annotations"),
        })
        .collect();

    // Build new parameters with expanded env types (e.g. `Print a` → `Print a [env_0]`).
    // This prevents `from_cst_type_no_type_variables` from auto-inserting fresh type variables.
    let expanded_parameters: Vec<cst::Parameter> = param_infos
        .iter()
        .enumerate()
        .map(|(i, (is_implicit, inner, typ))| {
            let env_name = context.names.push(Arc::new(format!("[env_{}]", i)));
            context.name_locations.push(location.clone());

            let expanded_type = add_env_to_trait_type(typ, env_name, &location);

            let new_pattern = context.patterns.push(Pattern::TypeAnnotation(*inner, expanded_type));
            assert_eq!(new_pattern, context.pattern_locations.push(location.clone()));

            cst::Parameter { is_implicit: *is_implicit, pattern: new_pattern }
        })
        .collect();

    if !impl_.trait_arguments.is_empty() || !impl_.parameters.is_empty() {
        let app_location = location.clone();
        let mut arguments = impl_.trait_arguments.clone();

        // Assume the returned trait captures each parameter.
        let parameter_types = expanded_parameters.iter().map(|param| match &context.patterns[param.pattern] {
            Pattern::TypeAnnotation(_, typ) => typ.clone(),
            _ => unreachable!("impl parameters are expected to have type annotations"),
        });
        arguments.push(make_tuple_type(&location, parameter_types));

        trait_type = Type::new(TypeKind::Application(Box::new(trait_type), arguments), app_location);
    }

    // If this is not a function we need to put the type annotation on the name itself rather than
    // the return type of the lambda.
    let pattern = if impl_.parameters.is_empty() {
        let pattern = context.patterns.push(Pattern::TypeAnnotation(variable, trait_type.clone()));
        assert_eq!(pattern, context.pattern_locations.push(location.clone()));
        pattern
    } else {
        variable
    };

    let fields = impl_.body.clone();
    let constructor = Expr::Constructor(Constructor { fields, typ: trait_type.clone() });
    let constructor = context.exprs.push(constructor);
    assert_eq!(constructor, context.expr_locations.push(location.clone()));

    let rhs = if impl_.parameters.is_empty() {
        constructor
    } else {
        let lambda = Expr::Lambda(Lambda {
            parameters: expanded_parameters,
            return_type: Some(trait_type),
            effects: Some(Vec::new()),
            body: constructor,
        });
        let lambda = context.exprs.push(lambda);
        assert_eq!(lambda, context.expr_locations.push(location));
        lambda
    };

    TopLevelItemKind::Definition(Definition { implicit: true, mutable: false, pattern, rhs })
}

/// Desugars
/// ```ante
/// trait Foo args with
///     declaration1: fn Arg1_1 ... ArgN_1 -> Ret_1
///     ...
///     declarationN: fn Arg1_N ... ArgN_N -> Ret_N
/// ```
/// Into
/// ```ante
/// type Foo args env =
///     declaration1: fn Arg1_1 ... ArgN_1 [env] -> Ret_1
///     ...
///     declarationN: fn Arg1_N ... ArgN_N [env] -> Ret_N
/// ```
fn desugar_trait(trait_: &TraitDefinition, context: &mut TopLevelContext) -> TopLevelItemKind {
    let name_location = context.name_locations[trait_.name].clone();

    // TODO: Can this be done more cleanly without resorting to strings users cannot type?
    let env = context.names.push(Arc::new("[env]".into()));
    context.name_locations.push(name_location.clone());

    // Add the `env` generic to the trait type itself
    let mut generics = trait_.generics.clone();
    generics.push(env);

    // Add `[env]` to each function type to make them implicitly closures
    let fields = mapvec(&trait_.body, |decl| {
        let typ = match &decl.typ.kind {
            cst::TypeKind::Function(f) => {
                let mut f = f.clone();
                f.environment = Some(Box::new(Type::new(TypeKind::Variable(env), name_location.clone())));
                Type::new(cst::TypeKind::Function(f), decl.typ.location.clone())
            },
            _ => decl.typ.clone(),
        };
        (decl.name, typ)
    });

    TopLevelItemKind::TypeDefinition(super::cst::TypeDefinition {
        shared: false,
        is_trait: true,
        name: trait_.name,
        generics,
        body: TypeDefinitionBody::Struct(fields),
    })
}

/// Traverse the expression recursively, desugaring along the way looking for the following cases:
/// - `loop`: All loops are sugar for an immediately invoked helper function
/// - `|>` and `<|`: Apply operators are sugar for direct application
/// - `foo _ x _`: Function calls with `_` as arguments are automatically converted into lambdas
///                with the `_` parameters as the remaining lambda parameters in source order.
fn desugar_expression(expr: ExprId, context: &mut TopLevelContext) {
    match &context.exprs[expr] {
        Expr::Error => (),
        Expr::Literal(_) => (),
        Expr::Variable(_) => (),
        Expr::Quoted(_) => (),
        Expr::Sequence(sequence) => todo!(),
        Expr::Definition(definition) => todo!(),
        Expr::MemberAccess(access) => todo!(),
        Expr::Call(call) => todo!(),
        Expr::Lambda(lambda) => todo!(),
        Expr::If(_) => todo!(),
        Expr::Match(_) => todo!(),
        Expr::Handle(handle) => todo!(),
        Expr::Reference(reference) => todo!(),
        Expr::TypeAnnotation(type_annotation) => todo!(),
        Expr::Constructor(constructor) => todo!(),
        Expr::Loop(_) => todo!(),
    }
}
