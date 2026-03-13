use std::num::NonZeroUsize;

/// A type's [Kind] is essentially the type of a type.
/// These differentiate whether something in a type position is itself
/// a type, a type constructor, or a type-level integer.
#[derive(Debug, PartialEq, Eq)]
pub enum Kind {
    /// A type valid in a type position
    Type,

    /// A type constructor expecting to be applied to N arguments,
    /// each of [Kind::Type]. This isn't required but is separated
    /// from [Kind::TypeConstructorComplex] to avoid allocation in the common case.
    TypeConstructorSimple(NonZeroUsize),

    /// A type constructor expecting to be applied to N arbitrary arguments.
    /// It is not an explicit requirement for this type, but at least one
    /// argument is expected to not be a [Kind::Type], since otherwise
    /// [Kind::TypeConstructorSimple] can be used which avoids an allocation.
    ///
    /// Requires the Vec of parameters to be non-empty.
    TypeConstructorComplex(Vec<Kind>),

    /// A trait constructor with the given parameters plus one additional
    /// optional closure environment parameter which is implicitly added.
    TraitConstructor(Vec<Kind>),

    /// A type-level `U32` used (for example) as an array length.
    U32,
}

impl Kind {
    pub fn accepts_arguments(&self, args: &[Kind]) -> bool {
        match self {
            Kind::Type => args.is_empty(),
            Kind::TypeConstructorSimple(expected) => {
                args.len() == usize::from(*expected) && args.iter().all(|kind| matches!(kind, Kind::Type))
            },
            Kind::TypeConstructorComplex(kinds) => kinds == args,
            Kind::TraitConstructor(kinds) => {
                if args.len() == kinds.len() + 1 {
                    kinds == &args[0..args.len() - 1]
                        // The optional env arg should be a Type
                        && matches!(&args[args.len() - 1], Kind::Type)
                } else {
                    kinds == args
                }
            },
            Kind::U32 => todo!(),
        }
    }
}
