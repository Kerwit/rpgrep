//! Verifica que la indexación poda directorios vendoreados / de build y que el
//! índice solo contiene el código fuente del proyecto.

use std::fs;

use rpgrep::chunk::Chunk;
use rpgrep::index::store::IndexStore;

#[test]
fn index_skips_node_modules_and_build_dirs() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();

    let mk = |rel: &str, body: &str| {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, body).unwrap();
    };

    mk(
        "src/auth/Login.ts",
        "export function login() { return authenticate(); }",
    );
    mk(
        "node_modules/react/index.js",
        "module.exports = function login() {}",
    );
    mk(
        "node_modules/typescript/lib.d.ts",
        "declare function login(): void;",
    );
    mk("dist/bundle.js", "function login(){return 0;}");
    mk("target/debug/x.rs", "fn login() {}");

    let store = IndexStore::from_dir(root, &["ts", "tsx", "js", "rs"], 40, 8).unwrap();

    assert!(!store.chunks.is_empty(), "debería indexar al menos src/");
    // Normaliza el separador para que el test sea agnóstico al SO (Windows usa `\`).
    let norm = |c: &Chunk| c.file.replace('\\', "/");
    for c in &store.chunks {
        let file = norm(c);
        assert!(
            !file.contains("node_modules")
                && !file.contains("/dist/")
                && !file.contains("/target/"),
            "indexó un fichero de un directorio excluido: {file}"
        );
    }
    assert!(
        store
            .chunks
            .iter()
            .any(|c| norm(c).contains("src/auth/Login.ts")),
        "debería haber indexado el fuente real src/auth/Login.ts"
    );
}
