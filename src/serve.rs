//! Servidor Unix-socket que sirve queries contra un `SearchPipeline`
//! cargado en memoria.
//!
//! Protocolo de cable (JSON-line):
//!
//! ```text
//!   client → server : {"query": "...", "budget": 4000, "topk": 50}\n
//!   server → client : {"results": [{"file", "start_line", "end_line",
//!                                    "score", "text"}, ...]}\n
//!                  | {"error": "..."}\n
//! ```
//!
//! Una conexión = una request = una response = cierre. Para alta QPS
//! basta con abrir varias conexiones (cada una se atiende en su propio
//! thread; el `SearchPipeline` se comparte como `Arc` y sus `search`
//! son `&self` → concurrencia de sólo-lectura segura).
//!
//! Sin hot-reload del índice (deferred): si `rpgrep watch` actualiza
//! el `.idx`, hay que reiniciar `serve` para tomarlo. La razón es
//! mantener el binario pequeño en v0.2.x; una próxima iteración puede
//! añadir un `{"reload": true}` en el protocolo o un signal handler.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;

use serde::{Deserialize, Serialize};

use crate::{Result, RpgrepError, SearchPipeline};

#[derive(Deserialize)]
struct Request {
    query: String,
    #[serde(default = "default_budget")]
    budget: usize,
    #[serde(default = "default_topk")]
    topk: usize,
}

fn default_budget() -> usize {
    4000
}
fn default_topk() -> usize {
    50
}

#[derive(Serialize)]
struct ResponseOk {
    results: Vec<HitJson>,
}

#[derive(Serialize)]
struct ResponseErr<'a> {
    error: &'a str,
}

#[derive(Serialize)]
struct HitJson {
    file: String,
    start_line: usize,
    end_line: usize,
    score: f32,
    text: String,
}

/// Bind del socket + accept loop. Bloquea hasta que el listener
/// muere (drop, error de accept persistente, o SIGINT).
pub fn run(index: &Path, socket: &Path) -> Result<()> {
    let pipeline = Arc::new(SearchPipeline::load(index)?);
    bind_and_serve(pipeline, socket)
}

fn bind_and_serve(pipeline: Arc<SearchPipeline>, socket: &Path) -> Result<()> {
    // Limpia socket previo (procesos anteriores no lo borran al salir).
    if socket.exists() {
        std::fs::remove_file(socket).map_err(|e| {
            RpgrepError::Persist(format!(
                "no se pudo limpiar el socket previo {}: {e}",
                socket.display()
            ))
        })?;
    }
    if let Some(parent) = socket.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let listener = UnixListener::bind(socket)
        .map_err(|e| RpgrepError::Persist(format!("bind {}: {e}", socket.display())))?;
    eprintln!("[rpgrep serve] listo en {}", socket.display());

    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                let pipe = Arc::clone(&pipeline);
                thread::spawn(move || {
                    if let Err(e) = handle_connection(stream, &pipe) {
                        eprintln!("[rpgrep serve] connection: {e:?}");
                    }
                });
            }
            Err(e) => eprintln!("[rpgrep serve] accept: {e:?}"),
        }
    }

    Ok(())
}

fn handle_connection(stream: UnixStream, pipeline: &SearchPipeline) -> Result<()> {
    let reader_stream = stream.try_clone()?;
    let mut reader = BufReader::new(reader_stream);
    let mut writer = stream;

    let mut line = String::new();
    let n = reader.read_line(&mut line)?;
    if n == 0 {
        return Ok(()); // cliente cerró sin enviar nada
    }

    let req: Request = match serde_json::from_str(&line) {
        Ok(r) => r,
        Err(e) => {
            return write_error(&mut writer, &format!("JSON inválido: {e}"));
        }
    };

    match pipeline.search(&req.query, req.budget, req.topk) {
        Ok(hits) => {
            let resp = ResponseOk {
                results: hits
                    .into_iter()
                    .map(|h| HitJson {
                        file: h.chunk.file,
                        start_line: h.chunk.start_line,
                        end_line: h.chunk.end_line,
                        score: h.score,
                        text: h.chunk.text,
                    })
                    .collect(),
            };
            let bytes = serde_json::to_vec(&resp)
                .map_err(|e| RpgrepError::Persist(format!("serialize response: {e}")))?;
            writer.write_all(&bytes)?;
            writer.write_all(b"\n")?;
        }
        Err(e) => {
            write_error(&mut writer, &format!("{e}"))?;
        }
    }

    Ok(())
}

fn write_error(writer: &mut UnixStream, msg: &str) -> Result<()> {
    let resp = ResponseErr { error: msg };
    let bytes = serde_json::to_vec(&resp)
        .map_err(|e| RpgrepError::Persist(format!("serialize error response: {e}")))?;
    writer.write_all(&bytes)?;
    writer.write_all(b"\n")?;
    Ok(())
}

/// Path por defecto del socket: `<index>/rpgrep.sock`. Convive con
/// `rpgrep.idx` en el mismo directorio para que un solo `--out` del
/// `watch` baste para localizar ambos.
pub fn default_socket_path(index: &Path) -> PathBuf {
    index.join("rpgrep.sock")
}
