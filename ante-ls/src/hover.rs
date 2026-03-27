use ante::incremental::{Db, GetItem, Parse, TypeCheck};
use ante::name_resolution::namespace::SourceFileId;

/// Find the innermost node (path, name, or pattern) at `byte_offset` in
/// `file_id` and return a hover string of the form `name : Type`.
///
/// Position lookups are done against the **desugared** context from `GetItem`
/// rather than the raw parse result, because type-checking runs on the
/// desugared form and the node IDs must match.
pub fn hover_at(compiler: &Db, file_id: SourceFileId, byte_offset: usize) -> Option<String> {
    use ante::parser::cst::Pattern;
    use ante::parser::ids::{NameId, PathId, PatternId};

    enum Id {
        Path(PathId, ante::parser::ids::TopLevelId),
        Name(NameId, ante::parser::ids::TopLevelId),
        Pattern(PatternId, ante::parser::ids::TopLevelId),
    }

    let parse = Parse(file_id).get(compiler);

    let mut best_span_len = usize::MAX;
    let mut best: Option<Id> = None;

    for item in &parse.cst.top_level_items {
        // Use the desugared context so node IDs match what TypeCheck stored.
        let (_, ctx) = GetItem(item.id).get(compiler);

        let mut check = |start: usize, end: usize, make_best: Id| {
            if start <= byte_offset && byte_offset < end {
                let span_len = end - start;
                if span_len < best_span_len {
                    best_span_len = span_len;
                    best = Some(make_best);
                }
            }
        };

        for (path_id, loc) in ctx.path_locations.iter() {
            check(loc.span.start.byte_index, loc.span.end.byte_index, Id::Path(path_id, item.id));
        }
        for (name_id, loc) in ctx.name_locations.iter() {
            check(loc.span.start.byte_index, loc.span.end.byte_index, Id::Name(name_id, item.id));
        }
        for (pattern_id, loc) in ctx.pattern_locations.iter() {
            check(loc.span.start.byte_index, loc.span.end.byte_index, Id::Pattern(pattern_id, item.id));
        }
    }

    match best? {
        Id::Path(path_id, item_id) => {
            let (_, ctx) = GetItem(item_id).get(compiler);
            let tc = TypeCheck(item_id).get(compiler);
            let typ = tc.result.maps.path_types.get(&path_id)?.follow(&tc.bindings);
            if is_sentinel(typ) {
                return None;
            }
            let name = ctx.paths.get(path_id)?.last_ident().to_owned();
            let type_str = typ.to_string(&tc.bindings, &tc.result.context, compiler);
            Some(format!("{name} : {type_str}"))
        },
        Id::Name(name_id, item_id) => {
            let (_, ctx) = GetItem(item_id).get(compiler);
            let tc = TypeCheck(item_id).get(compiler);
            let typ = tc.result.maps.name_types.get(&name_id)?.follow(&tc.bindings);
            if is_sentinel(typ) {
                return None;
            }
            let name = ctx.names.get(name_id).map(|n| n.as_str()).unwrap_or("_");
            let type_str = typ.to_string(&tc.bindings, &tc.result.context, compiler);
            Some(format!("{name} : {type_str}"))
        },
        Id::Pattern(pattern_id, item_id) => {
            let (_, ctx) = GetItem(item_id).get(compiler);
            let tc = TypeCheck(item_id).get(compiler);
            let typ = tc.result.maps.pattern_types.get(&pattern_id)?.follow(&tc.bindings);
            if is_sentinel(typ) {
                return None;
            }
            // Extract the variable name from Pattern::Variable; skip other pattern kinds.
            let name = match ctx.patterns.get(pattern_id)? {
                Pattern::Variable(name_id) => ctx.names.get(*name_id)?.as_str().to_owned(),
                _ => return None,
            };
            let type_str = typ.to_string(&tc.bindings, &tc.result.context, compiler);
            Some(format!("{name} : {type_str}"))
        },
    }
}

fn is_sentinel(typ: &ante::type_inference::types::Type) -> bool {
    use ante::type_inference::types::PrimitiveType;
    matches!(typ, ante::type_inference::types::Type::Primitive(PrimitiveType::Error | PrimitiveType::NoClosureEnv))
}
