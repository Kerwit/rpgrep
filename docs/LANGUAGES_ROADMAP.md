# Roadmap v0.2.6 — Registry de lenguajes + mainstream restantes

> **Estado**: ⏳ pendiente. La **parte lenguajes shipó en v0.2.5** (TypeScript,
> TSX, Dart, C integrados sobre el enum `Language`). Lo que queda aquí para
> v0.2.6: el refactor a `LangSpec` registry + feature flags y los mainstream
> restantes (Go, C++, Java). No ejecutar sin petición explícita.
> No ejecutar el refactor sin petición explícita.
> Referencia cruzada: `BLUEPRINT.md` §R7 y `SUMMARIES.md` "Deuda heredada (c)".

## 1. Motivación

`chunk_file` cubre hoy **7 lenguajes** (Rust, Python, JavaScript,
TypeScript, TSX, Dart, C) con AST chunking; el resto cae a `chunk_lines`
line-based. La ganancia medida del AST sobre line-based es **+24% Recall@5**
y **+8% Diversity@5** (ver `docs/VALIDATION.md` §5). Por cada lenguaje
mainstream que sigamos cortando por líneas, perdemos ese diferencial sobre
repos reales.

Objetivo v0.2.6: completar cobertura AST hacia ~10 lenguajes (faltan Go,
C++, Java) con coste marginal mínimo y sin inflar el binario por defecto.

## 2. Patrón actual y su límite

`src/chunk/mod.rs::Language` es un enum con **7 variants**. Añadir un lenguaje
requiere tocar **3 match arms** (`from_path`, `ts_language`,
`top_level_node_kinds`) además de la dep en `Cargo.toml`. Llegar a 10
lenguajes genera ~30 match arms con duplicación estructural — de ahí el
refactor a registry de la §3 (aún pendiente).

## 3. Plan de refactor

### 3.1. Registry (`src/chunk/lang.rs` nuevo)

```rust
use tree_sitter::Language as TsLanguage;

pub struct LangSpec {
    pub name: &'static str,
    pub extensions: &'static [&'static str],
    pub grammar: fn() -> TsLanguage,
    pub top_level_kinds: &'static [&'static str],
}

pub const REGISTRY: &[LangSpec] = &[
    #[cfg(feature = "lang-rust")]
    LangSpec {
        name: "rust",
        extensions: &["rs"],
        grammar: || tree_sitter_rust::language(),
        top_level_kinds: &[
            "function_item", "struct_item", "enum_item", "impl_item",
            "trait_item", "mod_item", "const_item", "static_item",
            "macro_definition", "type_item",
        ],
    },
    // … un bloque por lenguaje
];

pub fn detect(path: &Path) -> Option<&'static LangSpec> {
    let ext = path.extension()?.to_str()?.to_lowercase();
    REGISTRY
        .iter()
        .find(|l| l.extensions.iter().any(|e| *e == ext))
}
```

`chunk_with_ast` pasa a recibir `&LangSpec` en lugar de `Language`. El
enum `Language` desaparece.

### 3.2. Feature flags por lenguaje en `Cargo.toml`

```toml
[features]
default = ["lang-rust", "lang-python", "lang-javascript"]

lang-rust       = ["dep:tree-sitter-rust"]
lang-python     = ["dep:tree-sitter-python"]
lang-javascript = ["dep:tree-sitter-javascript"]
lang-go         = ["dep:tree-sitter-go"]
lang-typescript = ["dep:tree-sitter-typescript"]
lang-c          = ["dep:tree-sitter-c"]
lang-cpp        = ["dep:tree-sitter-cpp"]
lang-java       = ["dep:tree-sitter-java"]
lang-dart       = ["dep:tree-sitter-dart"]
lang-ruby       = ["dep:tree-sitter-ruby"]
lang-kotlin     = ["dep:tree-sitter-kotlin"]
lang-swift      = ["dep:tree-sitter-swift"]
lang-php        = ["dep:tree-sitter-php"]
lang-csharp     = ["dep:tree-sitter-c-sharp"]

all-languages = [
    "lang-rust", "lang-python", "lang-javascript",
    "lang-go", "lang-typescript", "lang-c", "lang-cpp", "lang-java",
    "lang-dart", "lang-ruby", "lang-kotlin", "lang-swift",
    "lang-php", "lang-csharp",
]

[dependencies]
tree-sitter            = "0.22"
tree-sitter-rust       = { version = "0.21", optional = true }
tree-sitter-python     = { version = "0.21", optional = true }
tree-sitter-javascript = { version = "0.21", optional = true }
tree-sitter-go         = { version = "0.21", optional = true }
# … resto
```

Resultado: `cargo build` (default) = ~10 MB con 3 lenguajes;
`cargo build --features all-languages` = ~15 MB con 10+.

## 4. Tabla de node kinds por lenguaje (Tier 1 + Tier 2)

| Lenguaje | Crate | `top_level_node_kinds` |
|---|---|---|
| **Tier 1 — mainstream** | | |
| Rust | `tree-sitter-rust` | `function_item`, `struct_item`, `enum_item`, `impl_item`, `trait_item`, `mod_item`, `const_item`, `static_item`, `macro_definition`, `type_item` |
| Python | `tree-sitter-python` | `function_definition`, `class_definition`, `decorated_definition` |
| JavaScript | `tree-sitter-javascript` | `function_declaration`, `class_declaration`, `method_definition`, `generator_function_declaration`, `lexical_declaration`, `export_statement` |
| **Go** | `tree-sitter-go` | `function_declaration`, `method_declaration`, `type_declaration`, `var_declaration`, `const_declaration` |
| **TypeScript** | `tree-sitter-typescript` | `function_declaration`, `class_declaration`, `interface_declaration`, `type_alias_declaration`, `enum_declaration`, `method_definition` |
| **C** | `tree-sitter-c` | `function_definition`, `struct_specifier`, `enum_specifier`, `declaration` (filtrar a top-level) |
| **C++** | `tree-sitter-cpp` | `function_definition`, `class_specifier`, `struct_specifier`, `enum_specifier`, `namespace_definition`, `template_declaration` |
| **Java** | `tree-sitter-java` | `class_declaration`, `interface_declaration`, `enum_declaration`, `method_declaration`, `record_declaration` |
| **Tier 2 — comunitarios** | | |
| Dart | `tree-sitter-dart` (verificar mantenimiento) | `function_signature`, `class_definition`, `enum_declaration`, `mixin_declaration`, `extension_declaration`, `top_level_variable_declaration` |
| Ruby | `tree-sitter-ruby` | `class`, `module`, `method`, `singleton_method` |
| Kotlin | `tree-sitter-kotlin` | `function_declaration`, `class_declaration`, `object_declaration` |
| Swift | `tree-sitter-swift` | `function_declaration`, `class_declaration`, `struct_declaration`, `enum_declaration`, `protocol_declaration` |
| PHP | `tree-sitter-php` | `function_definition`, `class_declaration`, `interface_declaration`, `trait_declaration` |
| C# | `tree-sitter-c-sharp` | `method_declaration`, `class_declaration`, `interface_declaration`, `struct_declaration`, `record_declaration` |

**Verificación de kinds**: para cada lenguaje, antes de mergear correr
`tree-sitter parse <ejemplo>` sobre un archivo representativo y
confirmar que los nombres en `node-types.json` del crate coinciden.
Equivocarse aquí es silencioso: el chunker devuelve 0 nodos → fallback
line-based sin error visible.

## 5. Checklist por lenguaje

1. ☐ Crate disponible y mantenido (último commit < 12 meses, versión `>= 0.20`).
2. ☐ Licencia compatible con MIT/Apache-2 (dual del proyecto).
3. ☐ Versión core `tree-sitter` consistente (`0.22.x` actualmente).
4. ☐ `top_level_node_kinds` verificados con `tree-sitter parse` o `node-types.json`.
5. ☐ Test `ast_<lang>_extracts_top_level_items_intact` (copiar plantilla de Rust).
6. ☐ Feature flag declarado en `Cargo.toml`.
7. ☐ Entrada en `REGISTRY` con `#[cfg(feature = "lang-<x>")]`.
8. ☐ Actualizar `SUMMARIES.md` (sección Configuración + Segmentación).

## 6. Coste estimado

- **Refactor a `LangSpec` registry**: ~30 min.
- **Tier 1 completo (4 lenguajes nuevos + tests)**: ~90 min.
- **Tier 2 selectivo (3-4 lenguajes)**: ~60 min adicionales si las grammars están vigentes.
- **Total v0.2.6 (refactor + Tier 1 restante)**: **~2 horas**.

## 7. Riesgos

- **Drift de grammars**: tree-sitter parsers cambian estructura entre
  versiones menores. Pin estricto con `=0.21.x` mitiga (no caret) para
  grammars comunitarias del Tier 2.
- **Tamaño binario por defecto**: feature flags lo mitigan. El default
  se mantiene en 3 lenguajes; el usuario que quiere todos pide
  `--features all-languages` explícito.
- **Tests AST inflados**: 14 tests AST (uno por lenguaje) cargan los 14
  parsers en el binario de tests = ~20-30 MB. Aceptable. Si molesta,
  mover los tests por lenguaje a su propio `#[cfg(feature = "lang-<x>")]`.

## 8. Cuándo abrir este frente

Cuando se cumpla **cualquiera** de:

- Un usuario pide soporte para un lenguaje específico (ya hay señal).
- El recall en repos con mix Go/Java/TS cae visiblemente vs Rust puros.
- Se publica una integración (Claude Code agent, VSCode ext) que necesita
  cobertura ≥80% de repos GitHub.

Hasta entonces, el fallback line-based es honesto y mantiene R3
(zero false negatives del Xor pre-screen sigue intacto).
