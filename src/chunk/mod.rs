//! Segmentación de archivos en unidades semánticas.
//!
//! Estrategia v0.2.4 con AST:
//!
//! - **AST chunking (preferido)** vía `tree-sitter`: cada nodo top-level
//!   funcional (fn/struct/impl/enum/trait/mod en Rust; def/class en
//!   Python; function/class/method en JS) se convierte en un chunk
//!   íntegro. Un `fn` no se corta a mitad. Excepción R1 documentada en
//!   BLUEPRINT — parsers determinísticos generados desde DSL, no ML.
//! - **Line-based (fallback)** para archivos sin parser disponible o
//!   con cero nodos top-level reconocibles (e.g., .md, .txt, scripts
//!   sin funciones declaradas). Mantiene el comportamiento v0.1 de
//!   ventanas de líneas con solapamiento.
//!
//! R4 intacta: `chunk_id = hash(path + start_line)`. La fórmula no
//! cambia; cambian los `start_line` porque el chunking ahora respeta
//! límites sintácticos en lugar de offsets line-based.

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;

#[derive(
    Debug, Clone, Serialize, Deserialize, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct Chunk {
    pub id: u64,
    /// Ruta del archivo origen en forma de string (rkyv no archiva
    /// `PathBuf` directamente en v0.8; `chunk_id` sigue hasheando sobre
    /// `&Path` así que la estabilidad de IDs no se ve afectada).
    pub file: String,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
}

impl Chunk {
    /// Estimación rápida de tokens (~4 chars/token para código).
    pub fn token_estimate(&self) -> usize {
        (self.text.len() / 4).max(1)
    }
}

/// Lenguajes con parser AST configurado. Detección por extensión.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Language {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Tsx,
    Dart,
    C,
}

impl Language {
    fn from_path(path: &Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?.to_lowercase();
        match ext.as_str() {
            "rs" => Some(Language::Rust),
            "py" => Some(Language::Python),
            "js" | "mjs" | "cjs" | "jsx" => Some(Language::JavaScript),
            "ts" | "mts" | "cts" => Some(Language::TypeScript),
            "tsx" => Some(Language::Tsx),
            "dart" => Some(Language::Dart),
            "c" | "h" => Some(Language::C),
            _ => None,
        }
    }

    fn ts_language(self) -> tree_sitter::Language {
        match self {
            Language::Rust => tree_sitter_rust::LANGUAGE.into(),
            Language::Python => tree_sitter_python::LANGUAGE.into(),
            Language::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            Language::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Language::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
            Language::Dart => tree_sitter_dart::LANGUAGE.into(),
            Language::C => tree_sitter_c::LANGUAGE.into(),
        }
    }

    /// Tipos de nodo AST que consideramos "chunks naturales". La lista
    /// está sesgada hacia *unidades funcionales completas* que el LLM
    /// va a consumir de una sentada: funciones, structs, métodos. No
    /// usamos imports, statements sueltos ni macros — esos viajan como
    /// parte del chunk envolvente cuando aplica.
    fn top_level_node_kinds(self) -> &'static [&'static str] {
        match self {
            Language::Rust => &[
                "function_item",
                "struct_item",
                "enum_item",
                "impl_item",
                "trait_item",
                "mod_item",
                "const_item",
                "static_item",
                "macro_definition",
                "type_item",
            ],
            Language::Python => &[
                "function_definition",
                "class_definition",
                "decorated_definition", // captura `@decorator\ndef ...` íntegro
            ],
            Language::JavaScript => &[
                "function_declaration",
                "class_declaration",
                "method_definition",
                "generator_function_declaration",
                "lexical_declaration", // `const foo = () => {}` top-level
                "export_statement",
            ],
            Language::TypeScript | Language::Tsx => &[
                "function_declaration",
                "generator_function_declaration",
                "class_declaration",
                "abstract_class_declaration",
                "method_definition",
                "interface_declaration",
                "type_alias_declaration",
                "enum_declaration",
                "internal_module", // `namespace Foo { … }`
                "lexical_declaration",
                "export_statement",
            ],
            Language::C => &[
                "function_definition",
                "struct_specifier",
                "enum_specifier",
                "union_specifier",
                "type_definition",      // `typedef …`
                "declaration",          // prototipos y globales top-level
                "preproc_function_def", // macros tipo función `#define f(x) …`
            ],
            Language::Dart => &[
                "class_declaration",
                "function_declaration",
                "mixin_declaration",
                "enum_declaration",
                "extension_declaration",
                "getter_declaration", // accessors top-level (`int get total => …`)
                "setter_declaration",
                "type_alias",                     // `typedef IntList = List<int>;`
                "top_level_variable_declaration", // const/final/var de nivel superior
            ],
        }
    }
}

/// Divide un archivo en chunks. Intenta AST chunking primero; si el
/// archivo no tiene parser configurado o el parser no produce nodos
/// top-level reconocidos, cae a `chunk_file_line_based`.
pub fn chunk_file(
    path: &Path,
    lines_per_chunk: usize,
    overlap: usize,
) -> std::io::Result<Vec<Chunk>> {
    let content = std::fs::read_to_string(path)?;
    if content.is_empty() {
        return Ok(vec![]);
    }

    if let Some(lang) = Language::from_path(path) {
        if let Some(ast_chunks) = chunk_with_ast(path, &content, lang) {
            if !ast_chunks.is_empty() {
                return Ok(ast_chunks);
            }
        }
    }

    Ok(chunk_lines(path, &content, lines_per_chunk, overlap))
}

/// Versión explícitamente AST-only: devuelve `None` si no hay parser
/// para el archivo (la decisión de fallback queda en el caller).
/// Expuesto para pruebas y para callers que necesiten distinguir.
pub fn chunk_file_ast(path: &Path, content: &str) -> Option<Vec<Chunk>> {
    let lang = Language::from_path(path)?;
    chunk_with_ast(path, content, lang)
}

fn chunk_with_ast(path: &Path, content: &str, lang: Language) -> Option<Vec<Chunk>> {
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&lang.ts_language()).ok()?;
    let tree = parser.parse(content, None)?;
    let root = tree.root_node();

    let kinds = lang.top_level_node_kinds();
    let mut chunks: Vec<Chunk> = Vec::new();
    let bytes = content.as_bytes();

    let mut cursor = root.walk();
    for child in root.children(&mut cursor) {
        if !kinds.contains(&child.kind()) {
            continue;
        }
        let start_line = child.start_position().row + 1; // 1-indexed
        let end_line = child.end_position().row + 1;
        let start_byte = child.start_byte();
        let end_byte = child.end_byte().min(bytes.len());
        if end_byte <= start_byte {
            continue;
        }
        let text = match std::str::from_utf8(&bytes[start_byte..end_byte]) {
            Ok(s) => s.to_string(),
            Err(_) => continue, // archivo no UTF-8 dentro del nodo: descartar
        };
        chunks.push(Chunk {
            id: chunk_id(path, start_line.saturating_sub(1)),
            file: path.to_string_lossy().into_owned(),
            start_line,
            end_line,
            text,
        });
    }

    if chunks.is_empty() {
        return None;
    }
    Some(chunks)
}

fn chunk_lines(path: &Path, content: &str, lines_per_chunk: usize, overlap: usize) -> Vec<Chunk> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return vec![];
    }

    let mut chunks = Vec::new();
    let stride = lines_per_chunk.saturating_sub(overlap).max(1);

    let mut i = 0;
    while i < lines.len() {
        let end = (i + lines_per_chunk).min(lines.len());
        let text = lines[i..end].join("\n");
        chunks.push(Chunk {
            id: chunk_id(path, i),
            file: path.to_string_lossy().into_owned(),
            start_line: i + 1,
            end_line: end,
            text,
        });
        if end == lines.len() {
            break;
        }
        i += stride;
    }
    chunks
}

fn chunk_id(path: &Path, start: usize) -> u64 {
    let mut h = DefaultHasher::new();
    path.hash(&mut h);
    start.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn chunks_overlap_correctly() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        // Archivo .txt sin parser AST → cae a line-based.
        let path = f.path().with_extension("txt");
        let body: String = (1..=100).map(|i| format!("line {i}\n")).collect();
        std::fs::write(&path, &body).unwrap();
        let chunks = chunk_file(&path, 20, 5).unwrap();
        assert!(!chunks.is_empty());
        // Stride = 15 → al menos 6 chunks para 100 líneas
        assert!(chunks.len() >= 6);
        // El primer chunk arranca en línea 1
        assert_eq!(chunks[0].start_line, 1);
        // Asegura que el path con extensión exista al limpiar
        let _ = std::fs::remove_file(&path);
        let _ = f.flush();
    }

    #[test]
    fn ast_rust_extracts_top_level_items_intact() {
        let mut tmp = tempfile::NamedTempFile::with_suffix(".rs").unwrap();
        let body = r#"
fn alpha(x: u32) -> u32 {
    x + 1
}

struct Beta {
    name: String,
    age: u8,
}

impl Beta {
    fn greet(&self) -> String {
        format!("hi {}", self.name)
    }
}

fn gamma() {
    println!("done");
}
"#;
        tmp.write_all(body.as_bytes()).unwrap();

        let chunks = chunk_file(tmp.path(), 40, 8).unwrap();
        // 4 nodos top-level: fn alpha, struct Beta, impl Beta, fn gamma.
        assert_eq!(chunks.len(), 4, "esperaba 4 chunks AST, recibí {chunks:?}");

        // Verificar que cada chunk contiene un nodo completo (no corte).
        assert!(chunks[0].text.contains("fn alpha"));
        assert!(chunks[0].text.contains("x + 1"));
        assert!(chunks[1].text.contains("struct Beta"));
        assert!(chunks[1].text.contains("age: u8"));
        assert!(chunks[2].text.contains("impl Beta"));
        assert!(chunks[2].text.contains("greet"));
        assert!(chunks[3].text.contains("fn gamma"));

        // start_line debe ser estrictamente creciente.
        for w in chunks.windows(2) {
            assert!(
                w[0].start_line < w[1].start_line,
                "chunks no ordenados por start_line"
            );
        }
    }

    #[test]
    fn ast_python_keeps_class_and_decorated_def_whole() {
        let mut tmp = tempfile::NamedTempFile::with_suffix(".py").unwrap();
        let body = r#"
def lonely_function(x):
    return x * 2

class MyClass:
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1

@decorator
def wrapped_function():
    pass
"#;
        tmp.write_all(body.as_bytes()).unwrap();

        let chunks = chunk_file(tmp.path(), 40, 8).unwrap();
        // 3 top-level: def lonely_function, class MyClass, @decorator def wrapped_function.
        assert_eq!(chunks.len(), 3, "esperaba 3 chunks AST, recibí {chunks:?}");
        assert!(chunks[0].text.contains("def lonely_function"));
        assert!(chunks[1].text.contains("class MyClass"));
        assert!(chunks[1].text.contains("def increment")); // método entero dentro del class
        assert!(chunks[2].text.contains("@decorator"));
        assert!(chunks[2].text.contains("def wrapped_function"));
    }

    #[test]
    fn ast_javascript_extracts_function_and_class() {
        let mut tmp = tempfile::NamedTempFile::with_suffix(".js").unwrap();
        let body = r#"
function alpha(x) {
    return x + 1;
}

class Beta {
    constructor(name) {
        this.name = name;
    }
    greet() {
        return "hi " + this.name;
    }
}

const gamma = (x) => x * 2;
"#;
        tmp.write_all(body.as_bytes()).unwrap();

        let chunks = chunk_file(tmp.path(), 40, 8).unwrap();
        // 3 nodos top-level: function alpha, class Beta, const gamma (lexical_declaration).
        assert!(
            chunks.len() >= 3,
            "esperaba ≥3 chunks AST en JS, recibí {chunks:?}"
        );
        assert!(chunks.iter().any(|c| c.text.contains("function alpha")));
        assert!(chunks.iter().any(|c| c.text.contains("class Beta")));
        assert!(chunks.iter().any(|c| c.text.contains("greet()")));
    }

    #[test]
    fn ast_dart_extracts_class_function_and_enum() {
        let mut tmp = tempfile::NamedTempFile::with_suffix(".dart").unwrap();
        let body = r#"
import 'dart:async';

void main() {
    print('hello');
}

class Service {
    final int id;
    Service(this.id);
    void run() {}
}

enum Color { red, green, blue }

mixin Logger {}

extension StringX on String {
    bool get isBlank => trim().isEmpty;
}
"#;
        tmp.write_all(body.as_bytes()).unwrap();

        let chunks = chunk_file(tmp.path(), 40, 8).unwrap();
        // Top-level: function main, class Service, enum Color, mixin Logger,
        // extension StringX (import_or_export se excluye → no genera chunk).
        assert!(
            chunks.len() >= 5,
            "esperaba ≥5 chunks AST en Dart, recibí {chunks:?}"
        );
        assert!(chunks.iter().any(|c| c.text.contains("void main()")));
        assert!(chunks.iter().any(|c| c.text.contains("class Service")));
        assert!(chunks.iter().any(|c| c.text.contains("enum Color")));
        assert!(chunks.iter().any(|c| c.text.contains("extension StringX")));
        // El import no debe constituir un chunk propio.
        assert!(
            !chunks
                .iter()
                .any(|c| c.text.trim() == "import 'dart:async';"),
            "el import no debería ser un chunk AST"
        );
    }

    #[test]
    fn ast_falls_back_to_line_based_when_no_top_level_items() {
        let mut tmp = tempfile::NamedTempFile::with_suffix(".rs").unwrap();
        // Archivo con sólo comentarios y un statement suelto (sin nodos
        // top-level reconocidos como chunks). AST devolverá None → fallback.
        let body = "// just a comment\n// another\nuse std::io;\n";
        tmp.write_all(body.as_bytes()).unwrap();

        let chunks = chunk_file(tmp.path(), 40, 8).unwrap();
        // El fallback line-based produce al menos 1 chunk.
        assert!(!chunks.is_empty());
    }

    #[test]
    fn chunk_id_stable_under_ast_reindex() {
        // R4: re-procesar el mismo archivo debe dar los mismos IDs.
        let mut tmp = tempfile::NamedTempFile::with_suffix(".rs").unwrap();
        let body = "fn alpha() {}\nfn beta() {}\n";
        tmp.write_all(body.as_bytes()).unwrap();

        let a = chunk_file(tmp.path(), 40, 8).unwrap();
        let b = chunk_file(tmp.path(), 40, 8).unwrap();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.id, y.id, "chunk_id inestable entre re-indexaciones");
        }
    }
}
