use std::{
    collections::{BTreeSet, HashMap},
    sync::Arc,
};

use ante::{
    diagnostics::{Diagnostic as AnteDiagnostic, DiagnosticKind},
    find_files,
    incremental::{Db, GetCrateGraph, Parse, SourceFile, TargetPointerSize, TypeCheck},
    name_resolution::namespace::CrateId,
};

use dashmap::DashMap;
use ropey::Rope;
use tower_lsp::lsp_types::*;

use crate::util::byte_range_to_lsp_range;

/// One-time setup: pointer size + full crate graph scan (local files + stdlib).
pub fn init_db(db: &mut Db, starting_file: &std::path::Path) {
    TargetPointerSize.set(db, 8);
    find_files::populate_crates_and_files(db, &[starting_file.to_path_buf()]);
}

/// Incrementally update a single file's content. inc-complete invalidates only
/// the cached queries that depend on this input.
pub fn set_file_content(db: &mut Db, path: &std::path::Path, rope: &Rope) {
    let file_id = ante::name_resolution::namespace::SourceFileId::new_in_local_crate(path);
    file_id.set(db, Arc::new(SourceFile::new(Arc::new(path.to_path_buf()), rope.to_string())));
}

/// Walk all items in the local crate, run TypeCheck on each, and convert the
/// accumulated compiler diagnostics to LSP diagnostics grouped by file URI.
/// All local crate files start with an empty list so stale diagnostics are
/// cleared for any file that no longer has errors.
pub fn collect_lsp_diagnostics(
    compiler: &Db, current_uri: &Url, current_rope: &Rope, document_map: &DashMap<Url, Rope>,
) -> HashMap<Url, Vec<Diagnostic>> {
    let crates = GetCrateGraph.get(compiler);
    let Some(local_crate) = crates.get(&CrateId::LOCAL) else {
        return HashMap::from([(current_uri.clone(), Vec::new())]);
    };

    // Pre-seed ALL local crate files with empty lists so stale diagnostics
    // are cleared for any file that no longer has errors.
    let mut result: HashMap<Url, Vec<Diagnostic>> = local_crate
        .source_files
        .keys()
        .filter_map(|path| Url::from_file_path(path.as_ref()).ok())
        .map(|uri| (uri, Vec::new()))
        .collect();
    // Ensure current_uri is present even if not yet in the crate's file list.
    result.entry(current_uri.clone()).or_insert_with(Vec::new);

    // Using a BTreeSet here deduplicates the diagnostics returned by get_accumulated
    let mut all_diags = BTreeSet::new();
    for file_id in local_crate.source_files.values() {
        let parse = Parse(*file_id).get(compiler);
        for item in &parse.cst.top_level_items {
            all_diags.extend(compiler.get_accumulated(TypeCheck(item.id)));
        }
    }

    for diag in &all_diags {
        if let Some((uri, lsp_diag)) = to_lsp_diagnostic(diag, compiler, current_uri, current_rope, document_map) {
            result.entry(uri).or_default().push(lsp_diag);
        }
    }

    result
}

/// Convert a single compiler `Diagnostic` to an LSP `Diagnostic`, returning the
/// file URI it belongs to alongside it. Returns `None` if the location cannot be
/// mapped (e.g. the file path cannot be expressed as a URI).
fn to_lsp_diagnostic(
    diag: &AnteDiagnostic, compiler: &Db, current_uri: &Url, current_rope: &Rope, document_map: &DashMap<Url, Rope>,
) -> Option<(Url, Diagnostic)> {
    let loc = diag.location();
    let source_file = loc.file_id.get(compiler);

    let uri = Url::from_file_path(source_file.path.as_ref()).ok()?;

    let rope = rope_for_file(&uri, &source_file.contents, current_uri, current_rope, document_map);

    let range = byte_range_to_lsp_range(loc.span.start.byte_index, loc.span.end.byte_index, &rope).ok()?;

    let lsp_diag = Diagnostic {
        range,
        severity: Some(to_severity(diag.kind())),
        message: diag.message(),
        source: Some("ante-ls".to_string()),
        ..Default::default()
    };

    Some((uri, lsp_diag))
}

/// Return the in-memory rope for a file: the live rope for the file currently
/// being edited, a cached rope for other open files, or a rope built from the
/// on-disk content stored in the compiler database as a last resort.
pub fn rope_for_file(
    uri: &Url, disk_contents: &str, current_uri: &Url, current_rope: &Rope, document_map: &DashMap<Url, Rope>,
) -> Rope {
    if uri == current_uri {
        current_rope.clone()
    } else {
        document_map.get(uri).map(|r| r.clone()).unwrap_or_else(|| Rope::from_str(disk_contents))
    }
}

fn to_severity(kind: DiagnosticKind) -> DiagnosticSeverity {
    match kind {
        DiagnosticKind::Note => DiagnosticSeverity::HINT,
        DiagnosticKind::Warning => DiagnosticSeverity::WARNING,
        DiagnosticKind::Error => DiagnosticSeverity::ERROR,
    }
}
