use std::{collections::BTreeSet, sync::Arc};

use rustc_hash::FxHashSet;

use crate::{
    name_resolution::Origin,
    parser::{
        cst,
        ids::{ExprId, NameId, PathId, PatternId},
    },
    type_inference::{
        TypeChecker,
        errors::TypeErrorKind,
        types::{PrimitiveType, Type},
    },
};

impl TypeChecker<'_, '_> {
    /// Finds the environment type for the given lambda. This involves finding the free variables
    /// within the lambda. This will unify the given `expected_environment_type` with the actual
    /// environment type found but will not actually perform closure conversion. Closure conversion
    /// is instead done while building the initial [crate::mir::Mir].
    pub(super) fn check_for_closure(&mut self, id: ExprId, expected_environment_type: &Type) {
        let mut context = FreeVars::default();
        context.find_free_variables(id, self);

        let env_type = make_env_type_with_names(&context.free_vars, self);
        self.unify(&env_type, expected_environment_type, TypeErrorKind::ClosureEnv, id);

        if !context.free_vars.is_empty() {
            self.current_extended_context_mut().insert_closure_environment(id, context.free_vars);
        }
    }
}

#[derive(Default)]
struct FreeVars {
    /// The free variables found
    free_vars: BTreeSet<NameId>,

    // We don't care about different scopes within the function
    defined_in_fn: FxHashSet<NameId>,
}

impl FreeVars {
    fn find_free_variables(&mut self, expr: ExprId, checker: &TypeChecker) {
        match &checker.current_extended_context()[expr] {
            cst::Expr::Error => (),
            cst::Expr::Literal(_) => (),
            cst::Expr::Variable(path) => self.find_free_variable(*path, checker),
            cst::Expr::Sequence(items) => {
                for item in items {
                    self.find_free_variables(item.expr, checker);
                }
            },
            cst::Expr::Definition(definition) => {
                self.declare_pattern(definition.pattern, checker);
                self.find_free_variables(definition.rhs, checker);
            },
            cst::Expr::MemberAccess(access) => self.find_free_variables(access.object, checker),
            cst::Expr::Call(call) => {
                self.find_free_variables(call.function, checker);
                for argument in call.arguments.iter() {
                    self.find_free_variables(argument.expr, checker);
                }
            },
            cst::Expr::Lambda(lambda) => {
                for parameter in lambda.parameters.iter() {
                    self.declare_pattern(parameter.pattern, checker);
                }
                self.find_free_variables(lambda.body, checker);
            },
            cst::Expr::If(if_) => {
                self.find_free_variables(if_.condition, checker);
                self.find_free_variables(if_.then, checker);
                if let Some(else_) = if_.else_ {
                    self.find_free_variables(else_, checker);
                }
            },
            cst::Expr::Match(match_) => {
                self.find_free_variables(match_.expression, checker);
                for (pattern, branch) in match_.cases.iter() {
                    self.declare_pattern(*pattern, checker);
                    self.find_free_variables(*branch, checker);
                }
            },
            cst::Expr::Handle(handle) => {
                self.find_free_variables(handle.expression, checker);
                for (pattern, branch) in handle.cases.iter() {
                    for argument in pattern.args.iter() {
                        self.declare_pattern(*argument, checker);
                    }
                    self.find_free_variables(*branch, checker);
                }
            },
            cst::Expr::Reference(reference) => self.find_free_variables(reference.rhs, checker),
            cst::Expr::TypeAnnotation(annotation) => self.find_free_variables(annotation.lhs, checker),
            cst::Expr::Constructor(constructor) => {
                for (_name, expr) in constructor.fields.iter() {
                    self.find_free_variables(*expr, checker);
                }
            },
            cst::Expr::Quoted(_) => (),
        }
    }

    /// Inserts any [NameId]s of values within this [PatternId] into `self.defined_in_fn`
    fn declare_pattern(&mut self, pattern: PatternId, checker: &TypeChecker) {
        match &checker.current_extended_context()[pattern] {
            cst::Pattern::Error => (),
            cst::Pattern::Variable(name) => {
                self.defined_in_fn.insert(*name);
            },
            cst::Pattern::Literal(_) => (),
            cst::Pattern::Constructor(_, fields) => {
                for field in fields {
                    self.declare_pattern(*field, checker);
                }
            },
            cst::Pattern::TypeAnnotation(pattern, _) => self.declare_pattern(*pattern, checker),
            cst::Pattern::MethodName { type_name: _, item_name } => {
                self.defined_in_fn.insert(*item_name);
            },
        }
    }

    fn find_free_variable(&mut self, path: PathId, checker: &TypeChecker) {
        if let Some(Origin::Local(name)) = checker.path_origin(path) {
            self.check_name(name);
        }
    }

    fn check_name(&mut self, name: NameId) {
        if !self.defined_in_fn.contains(&name) {
            self.free_vars.insert(name);
        }
    }
}

fn make_env_type_with_names(free_vars: &BTreeSet<NameId>, checker: &TypeChecker) -> Type {
    let free_vars = free_vars.iter().map(|name| checker.name_types[name].clone());
    make_env_type(free_vars)
}

fn make_env_type(mut free_vars: impl ExactSizeIterator<Item = Type>) -> Type {
    let Some(free_var) = free_vars.next() else {
        return Type::Primitive(PrimitiveType::NoClosureEnv);
    };

    if free_vars.len() == 0 {
        free_var
    } else {
        let rest = make_env_type(free_vars);
        Type::Application(Arc::new(Type::PAIR), Arc::new(vec![free_var, rest]))
    }
}
