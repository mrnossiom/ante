use std::{
    collections::{BTreeSet, HashMap},
    env::set_current_dir,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use ante::{
    diagnostics::{Diagnostic as AnteDiagnostic, DiagnosticKind, Location as AnteLocation},
    find_files,
    incremental::{Db, GetCrateGraph, GetItem, Parse, Resolve, SourceFile, TargetPointerSize, TypeCheck},
    name_resolution::{namespace::{CrateId, SourceFileId}, Origin},
    type_inference::types::TypeBindings,
};

use dashmap::DashMap;
use futures::future::join_all;
use ropey::Rope;
use tokio::sync::RwLock;
use tower_lsp::{
    jsonrpc::{Error, Result},
    lsp_types::*,
    Client, LanguageServer, LspService, Server,
};

mod util;
use util::{byte_range_to_lsp_range, lsp_range_to_rope_range, position_to_byte_offset};

struct Backend {
    client: Client,
    document_map: DashMap<Url, Rope>,
    compiler: RwLock<Db>,
    db_initialized: AtomicBool,
}

// ── LSP protocol implementation ───────────────────────────────────────────────

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, params: InitializeParams) -> Result<InitializeResult> {
        self.client.log_message(MessageType::LOG, format!("ante-ls initialize: {:?}", params)).await;
        if let Some(root_uri) = params.root_uri {
            let root = PathBuf::from(root_uri.path());
            if set_current_dir(&root).is_err() {
                self.client
                    .log_message(MessageType::ERROR, format!("Failed to set root directory to {:?}", root))
                    .await;
            }
        }

        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::INCREMENTAL)),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                definition_provider: Some(OneOf::Left(true)),
                ..Default::default()
            },
            ..Default::default()
        })
    }

    async fn initialized(&self, params: InitializedParams) {
        self.client.log_message(MessageType::LOG, format!("ante-ls initialized: {:?}", params)).await;
    }

    async fn shutdown(&self) -> Result<()> {
        self.client.log_message(MessageType::LOG, "ante-ls shutdown".to_string()).await;
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        self.client.log_message(MessageType::LOG, format!("ante-ls did_open: {:?}", params.text_document.uri)).await;
        let rope = Rope::from_str(&params.text_document.text);
        self.document_map.insert(params.text_document.uri.clone(), rope.clone());
        self.update_diagnostics(params.text_document.uri, &rope).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        self.client.log_message(MessageType::LOG, format!("ante-ls did_change: {:?}", params.text_document.uri)).await;
        self.document_map.alter(&params.text_document.uri, |_, mut rope| {
            for change in params.content_changes {
                if let Some(range) = change.range {
                    if let Ok(range) = lsp_range_to_rope_range(range, &rope) {
                        rope.remove(range.clone());
                        rope.insert(range.start, &change.text);
                    }
                } else {
                    rope = Rope::from_str(&change.text);
                }
            }
            rope
        });
        if let Some(rope) = self.document_map.get(&params.text_document.uri) {
            self.update_diagnostics(params.text_document.uri, &rope).await;
        }
    }

    async fn did_save(&self, params: DidSaveTextDocumentParams) {
        self.client.log_message(MessageType::LOG, format!("ante-ls did_save: {:?}", params.text_document.uri)).await;
        if let Some(text) = params.text {
            let rope = Rope::from_str(&text);
            self.document_map.insert(params.text_document.uri.clone(), rope);
        }
        if let Some(rope) = self.document_map.get(&params.text_document.uri) {
            self.update_diagnostics(params.text_document.uri, &rope).await;
        }
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        let rope = match self.document_map.get(&uri) {
            Some(r) => r.clone(),
            None => return Ok(None),
        };
        let byte_offset = match position_to_byte_offset(position, &rope) {
            Some(b) => b,
            None => return Ok(None),
        };
        let path = match uri.to_file_path() {
            Ok(p) => p,
            Err(_) => return Ok(None),
        };
        let file_id = SourceFileId::new_in_local_crate(&path);

        let hover_text = {
            let compiler = self.compiler.read().await;
            hover_at(&compiler, file_id, byte_offset)
        };

        match hover_text {
            Some(value) => Ok(Some(Hover {
                contents: HoverContents::Markup(MarkupContent { kind: MarkupKind::PlainText, value }),
                range: None,
            })),
            None => Err(Error::method_not_found()),
        }
    }

    async fn goto_definition(&self, params: GotoDefinitionParams) -> Result<Option<GotoDefinitionResponse>> {
        let uri = params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        let rope = match self.document_map.get(&uri) {
            Some(r) => r.clone(),
            None => return Err(Error::method_not_found()),
        };
        let byte_offset = match position_to_byte_offset(position, &rope) {
            Some(b) => b,
            None => return Err(Error::method_not_found()),
        };
        let path = match uri.to_file_path() {
            Ok(p) => p,
            Err(_) => return Err(Error::method_not_found()),
        };
        let file_id = SourceFileId::new_in_local_crate(&path);

        let lsp_location = {
            let compiler = self.compiler.read().await;
            let ante_loc = match definition_at(&compiler, file_id, byte_offset) {
                Some(loc) => loc,
                None => return Err(Error::method_not_found()),
            };
            let source_file = ante_loc.file_id.get(&*compiler);
            let def_uri = match Url::from_file_path(source_file.path.as_ref()) {
                Ok(u) => u,
                Err(_) => return Err(Error::method_not_found()),
            };
            let def_rope = rope_for_file(&def_uri, &source_file.contents, &uri, &rope, &self.document_map);
            let range = match byte_range_to_lsp_range(
                ante_loc.span.start.byte_index,
                ante_loc.span.end.byte_index,
                &def_rope,
            ) {
                Ok(r) => r,
                Err(_) => return Err(Error::method_not_found()),
            };
            Location { uri: def_uri, range }
        };

        Ok(Some(GotoDefinitionResponse::Scalar(lsp_location)))
    }
}

// ── Compiler interaction ──────────────────────────────────────────────────────

impl Backend {
    /// Update the compiler database with the latest in-memory file content, then
    /// collect and publish diagnostics for the local crate.
    async fn update_diagnostics(&self, uri: Url, rope: &Rope) {
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

/// One-time setup: pointer size + full crate graph scan (local files + stdlib).
fn init_db(db: &mut Db, starting_file: &std::path::Path) {
    TargetPointerSize.set(db, 8);
    find_files::populate_crates_and_files(db, &[starting_file.to_path_buf()]);
}

/// Incrementally update a single file's content. inc-complete invalidates only
/// the cached queries that depend on this input.
fn set_file_content(db: &mut Db, path: &std::path::Path, rope: &Rope) {
    let file_id = SourceFileId::new_in_local_crate(path);
    file_id.set(db, Arc::new(SourceFile::new(Arc::new(path.to_path_buf()), rope.to_string())));
}

/// Walk all items in the local crate, run TypeCheck on each, and convert the
/// accumulated compiler diagnostics to LSP diagnostics grouped by file URI.
/// The current file starts with an empty list so stale diagnostics are cleared.
fn collect_lsp_diagnostics(
    compiler: &Db, current_uri: &Url, current_rope: &Rope, document_map: &DashMap<Url, Rope>,
) -> HashMap<Url, Vec<Diagnostic>> {
    let mut result: HashMap<Url, Vec<Diagnostic>> = HashMap::from([(current_uri.clone(), Vec::new())]);

    let crates = GetCrateGraph.get(compiler);
    let Some(local_crate) = crates.get(&CrateId::LOCAL) else {
        return result;
    };

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
fn rope_for_file<'a>(
    uri: &Url, disk_contents: &str, current_uri: &Url, current_rope: &'a Rope, document_map: &'a DashMap<Url, Rope>,
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

// ── Hover ─────────────────────────────────────────────────────────────────────

/// Find the innermost node (path, name, or pattern) at `byte_offset` in
/// `file_id` and return a hover string of the form `name : Type`.
///
/// Position lookups are done against the **desugared** context from `GetItem`
/// rather than the raw parse result, because type-checking runs on the
/// desugared form and the node IDs must match.
fn hover_at(compiler: &Db, file_id: SourceFileId, byte_offset: usize) -> Option<String> {
    use ante::parser::cst::Pattern;
    use ante::parser::ids::{NameId, PathId, PatternId};

    enum Best {
        Path(PathId, ante::parser::ids::TopLevelId),
        Name(NameId, ante::parser::ids::TopLevelId),
        Pattern(PatternId, ante::parser::ids::TopLevelId),
    }

    let parse = Parse(file_id).get(compiler);

    let mut best_span_len = usize::MAX;
    let mut best: Option<Best> = None;

    for item in &parse.cst.top_level_items {
        // Use the DESUGARED context so node IDs match what TypeCheck stored.
        let (_, ctx) = GetItem(item.id).get(compiler);

        let mut check = |start: usize, end: usize, make_best: Best| {
            if start <= byte_offset && byte_offset < end {
                let span_len = end - start;
                if span_len < best_span_len {
                    best_span_len = span_len;
                    best = Some(make_best);
                }
            }
        };

        for (path_id, loc) in ctx.path_locations.iter() {
            check(loc.span.start.byte_index, loc.span.end.byte_index, Best::Path(path_id, item.id));
        }
        for (name_id, loc) in ctx.name_locations.iter() {
            check(loc.span.start.byte_index, loc.span.end.byte_index, Best::Name(name_id, item.id));
        }
        for (pattern_id, loc) in ctx.pattern_locations.iter() {
            check(loc.span.start.byte_index, loc.span.end.byte_index, Best::Pattern(pattern_id, item.id));
        }
    }

    match best? {
        Best::Path(path_id, item_id) => {
            let (_, ctx) = GetItem(item_id).get(compiler);
            let tc = TypeCheck(item_id).get(compiler);
            let typ = tc.result.maps.path_types.get(&path_id)?.follow(&tc.bindings);
            if is_sentinel(typ) {
                return None;
            }
            let name = ctx.paths.get(path_id)?.last_ident().to_owned();
            let type_str = type_to_string(typ, &tc.bindings, &tc.result.context, compiler);
            Some(format!("{name} : {type_str}"))
        },
        Best::Name(name_id, item_id) => {
            let (_, ctx) = GetItem(item_id).get(compiler);
            let tc = TypeCheck(item_id).get(compiler);
            let typ = tc.result.maps.name_types.get(&name_id)?.follow(&tc.bindings);
            if is_sentinel(typ) {
                return None;
            }
            let name = ctx.names.get(name_id).map(|n| n.as_str()).unwrap_or("_");
            let type_str = type_to_string(typ, &tc.bindings, &tc.result.context, compiler);
            Some(format!("{name} : {type_str}"))
        },
        Best::Pattern(pattern_id, item_id) => {
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
            let type_str = type_to_string(typ, &tc.bindings, &tc.result.context, compiler);
            Some(format!("{name} : {type_str}"))
        },
    }
}

fn is_sentinel(typ: &ante::type_inference::types::Type) -> bool {
    use ante::type_inference::types::PrimitiveType;
    matches!(typ, ante::type_inference::types::Type::Primitive(PrimitiveType::Error | PrimitiveType::NoClosureEnv))
}

/// Thin wrapper so callers don't need to import all the generic bounds of
/// `Type::to_string`.
fn type_to_string(
    typ: &ante::type_inference::types::Type, bindings: &TypeBindings, names: &impl ante::parser::ids::NameStore,
    compiler: &Db,
) -> String {
    typ.to_string(bindings, names, compiler)
}

// ── Go-to-definition ──────────────────────────────────────────────────────────

/// Find the definition location of the symbol (path) under `byte_offset`.
///
/// Paths (variable uses, function calls) are looked up via the `Resolve` query
/// which maps each `PathId` to its `Origin`. The definition location is then
/// read from the **raw** parse context (not desugared), because that is what
/// `TopLevelName::location` uses and where source positions are stored.
///
/// Returns `None` for builtins, unresolved names, or when no path covers the
/// given byte offset.
fn definition_at(compiler: &Db, file_id: SourceFileId, byte_offset: usize) -> Option<AnteLocation> {
    use ante::parser::ids::PathId;

    let parse = Parse(file_id).get(compiler);

    let mut best_span_len = usize::MAX;
    let mut best: Option<(PathId, ante::parser::ids::TopLevelId)> = None;

    for item in &parse.cst.top_level_items {
        // Use desugared context: Resolve also operates on the desugared form,
        // so PathIds must come from the same source.
        let (_, ctx) = GetItem(item.id).get(compiler);
        for (path_id, loc) in ctx.path_locations.iter() {
            let start = loc.span.start.byte_index;
            let end = loc.span.end.byte_index;
            if start <= byte_offset && byte_offset < end && (end - start) < best_span_len {
                best_span_len = end - start;
                best = Some((path_id, item.id));
            }
        }
    }

    let (path_id, item_id) = best?;
    let resolve = Resolve(item_id).get(compiler);

    match *resolve.path_origins.get(&path_id)? {
        Origin::TopLevelDefinition(top_level_name) => {
            // Definition is in a (possibly different) top-level item's raw parse context.
            let def_parse = Parse(top_level_name.top_level_item.source_file).get(compiler);
            let def_ctx = def_parse.top_level_data.get(&top_level_name.top_level_item)?;
            def_ctx.name_locations.get(top_level_name.local_name_id).cloned()
        },
        Origin::Local(name_id) => {
            // Local binding (parameter, let-binding, etc.) in the same top-level item.
            let def_parse = Parse(item_id.source_file).get(compiler);
            let def_ctx = def_parse.top_level_data.get(&item_id)?;
            def_ctx.name_locations.get(name_id).cloned()
        },
        Origin::TypeResolution | Origin::Builtin(_) => None,
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    env_logger::init();

    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::build(|client| Backend {
        client,
        document_map: DashMap::new(),
        compiler: RwLock::new(Db::default()),
        db_initialized: AtomicBool::new(false),
    })
    .finish();

    Server::new(stdin, stdout, socket).serve(service).await;
}
