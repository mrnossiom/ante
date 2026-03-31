use std::{
    collections::HashMap,
    sync::{atomic::Ordering, Arc},
};

use ante::{
    diagnostics::{Diagnostic as AnteDiagnostic, DiagnosticKind},
    find_files,
    incremental::{AllDiagnostics, Db, SourceFile, TargetPointerSize},
};

use dashmap::DashMap;
use futures::future::join_all;
use ropey::Rope;
use tower_lsp::lsp_types::*;

use crate::{util::byte_range_to_lsp_range, Backend};

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

impl Backend {
    /// Update the compiler database with the latest in-memory file content, then
    /// collect and publish diagnostics for the local crate.
    pub(super) async fn update_diagnostics(&self, uri: Url, rope: &Rope) {
        let path = match uri.to_file_path() {
            Ok(p) => p,
            Err(_) => {
                self.client.log_message(MessageType::ERROR, format!("Failed to convert URI to path: {uri}")).await;
                return;
            },
        };

        // Write phase: initialize once, then update the changed file's content.
        {
            let mut compiler = self.compiler.write().await;
            if !self.db_initialized.swap(true, Ordering::SeqCst) {
                init_db(&mut compiler, &path);
            }
            set_file_content(&mut compiler, &path, rope);
        }

        // Read phase: collect diagnostics without blocking writers unnecessarily.
        let lsp_diagnostics = {
            let compiler = self.compiler.read().await;
            collect_lsp_diagnostics(&compiler, &uri, rope, &self.document_map)
        };

        join_all(lsp_diagnostics.into_iter().map(|(u, d)| self.client.publish_diagnostics(u, d, None))).await;
    }
}

/// Walk all items in the local crate, run TypeCheck on each, and convert the
/// accumulated compiler diagnostics to LSP diagnostics grouped by file URI.
/// All local crate files start with an empty list so stale diagnostics are
/// cleared for any file that no longer has errors.
pub fn collect_lsp_diagnostics(
    compiler: &Db, current_uri: &Url, current_rope: &Rope, document_map: &DashMap<Url, Rope>,
) -> HashMap<Url, Vec<Diagnostic>> {
    let diagnostics = AllDiagnostics.get(compiler);
    let mut url_to_diagnostics = HashMap::<_, Vec<_>>::default();

    for diagnostic in diagnostics.iter() {
        if let Some((uri, lsp_diag)) = to_lsp_diagnostic(diagnostic, compiler, current_uri, current_rope, document_map)
        {
            url_to_diagnostics.entry(uri).or_default().push(lsp_diag);
        }
    }

    url_to_diagnostics
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
