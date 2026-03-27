use std::sync::atomic::{AtomicBool, Ordering};

use ante::incremental::Db;

use dashmap::DashMap;
use futures::future::join_all;
use ropey::Rope;
use tokio::sync::RwLock;
use tower_lsp::{
    jsonrpc::{Error, Result},
    lsp_types::*,
    Client, LanguageServer, LspService, Server,
};

mod definition;
mod diagnostics;
mod hover;
mod util;

use definition::definition_at;
use diagnostics::{collect_lsp_diagnostics, init_db, rope_for_file, set_file_content};
use hover::hover_at;
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
            let root = std::path::PathBuf::from(root_uri.path());
            if std::env::set_current_dir(&root).is_err() {
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
        let file_id = ante::name_resolution::namespace::SourceFileId::new_in_local_crate(&path);

        let hover_text = {
            let compiler = self.compiler.read().await;
            hover_at(&compiler, file_id, byte_offset)
        };

        Ok(hover_text.map(|value| Hover {
            contents: HoverContents::Markup(MarkupContent { kind: MarkupKind::PlainText, value }),
            range: None,
        }))
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
        let file_id = ante::name_resolution::namespace::SourceFileId::new_in_local_crate(&path);

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
