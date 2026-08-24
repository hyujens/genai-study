# RAG Improvement Items

## Baseline

- Review scope: `rag/` implementation, tests, and package configuration.
- Baseline command: `uv run pytest -q`.
- Baseline result: `16 passed`.
- Goal: preserve the repository's learning-first structure while making concepts, intentional simplifications, and behavior easier to understand and verify.

## P1 — Correctness and behavioral consistency

### 1. Use one stable document ID strategy

Current behavior:

- `EmbeddingModel.generate()` falls back to `doc_{index}`.
- `EmbeddingModel.update()` falls back to the first 50 characters of content.
- This follows the reference example, but a document without `metadata["id"]` cannot be reliably recognized during a later incremental update.

Improvement:

- Centralize ID selection in one helper and use it in both `generate()` and `update()`.
- Prefer a deterministic content-derived ID if persistence across reordered inputs is required.

Acceptance criteria:

- A document without an explicit ID is embedded once and skipped on the next update.
- Reordering input documents does not change their fallback IDs.
- Unit tests cover explicit IDs, missing IDs, and falsy IDs.

### 2. Make `keep_separator` behavior explicit

Current behavior:

- `SplitConfig.keep_separator` is stored but none of the splitter implementations use it.

Improvement:

- Implement separator preservation, or document that the option is intentionally deferred and remove it from the active public configuration until implemented.

Acceptance criteria:

- The setting has observable, tested behavior, or is clearly marked as unavailable.
- Tests cover both enabled and disabled behavior if implemented.

### 3. Align token-based length accounting

Current behavior:

- `TokenTextSplitter` approximates each text segment with `ceil(len(text) / 4)`.
- `TextSplitter.merge_splits()` measures separators with character length.
- `encoding_name` is stored but does not select a tokenizer.

Improvement:

- Clearly label this as approximate token splitting.
- Either measure all chunk components consistently with the configured length function or document why separator characters remain character-counted to match the reference implementation.

Acceptance criteria:

- Class documentation states that no real tokenizer is used.
- Tests demonstrate the approximation and its chunk-size meaning.
- `encoding_name` is either used or explicitly documented as placeholder metadata.

### 4. Make PDF source handling URI-safe

Current behavior:

- Local paths are derived by removing the `file://` prefix as text.
- A URI produced by `Path.as_uri()` can contain percent-encoded characters.

Improvement:

- Parse file URIs and decode their path before passing it to PyMuPDF.
- Avoid replacing every occurrence of `file://` in the source string.

Acceptance criteria:

- A local PDF whose path contains spaces loads successfully.
- Tests cover a normal local path and a percent-encoded file URI.

## P2 — Learning clarity and API design

### 5. Add a learning-oriented `rag/README.md`

Current behavior:

- `rag/README.md` is empty.

Improvement:

- Describe the pipeline in repository order: load → clean → split → embed → retrieve.
- List the corresponding source examples used for each module.
- Explain the distinction between a source file, loaded `Document`, and chunk `Document`.
- List intentional simplifications such as keyword matching, approximate tokens, and JSON embedding storage.
- Include `uv run pytest` as the verification command.

Acceptance criteria:

- A returning reader can identify each module's responsibility and entry point from the README alone.
- Intentional deviations from the JavaScript reference are listed explicitly.

### 6. Introduce typed embedding records

Current behavior:

- Embedding records and storage results use unstructured `dict` and `list[dict]` annotations.

Improvement:

- Introduce a `TypedDict` or dataclass for an embedding record and JSON file schema.
- Keep JSON serialization straightforward and visible for learning purposes.

Acceptance criteria:

- `generate()`, `save_as_json()`, `load_from_file()`, lookup, and update share the same record type.
- Required fields and timestamp units are visible in the type definition.

### 7. Separate model location from model name

Current behavior:

- `EmbeddingModel` receives a model filename but constructs its path from a fixed directory under `Path.home()`.
- Tests bypass `__init__()` to avoid loading a real GGUF model.

Improvement:

- Accept a `Path` for the model file, or inject an embedding backend through a small protocol.
- Keep a convenience constructor for the repository's default model location if useful.

Acceptance criteria:

- Unit tests can construct `EmbeddingModel` without `__new__()`.
- The production path still enables `embedding=True` and preserves the current llama.cpp settings.

### 8. Clarify PDF loader lifecycle

Current behavior:

- `PDFLoader.__init__()` performs file/network I/O and keeps a PyMuPDF document open.

Improvement:

- Decide and document whether construction opens the source or `load()` owns I/O.
- Provide a clear close/context-manager lifecycle for the PyMuPDF document.

Acceptance criteria:

- Resource ownership is visible from the API.
- Tests confirm the loader can be used through its intended lifecycle.

## P3 — Test coverage

### 9. Expand keyword retrieval tests

Add tests for:

- `top_k` enforcement.
- no-match queries.
- stop-word-only queries.
- punctuation behavior, documenting whether token normalization is intentionally simple.
- ranking when documents have different scores.

Acceptance criteria:

- Tests capture every behavior previously diagnosed in `naive_keyword_search()`.

### 10. Add data cleaning and PDF loader tests

Add tests for:

- page marker removal.
- whitespace normalization while preserving paragraph breaks.
- split-page and whole-document modes.
- one-based page metadata and total page count.
- local file URI loading.
- HTTP success and failure using a mocked request rather than external network access.

Acceptance criteria:

- PDF tests use a small generated or checked-in fixture.
- The normal test suite does not access the network or the user's Downloads directory.

### 11. Add embedding storage and update cases

Add tests for:

- empty embedding collections and the documented dimension fallback.
- JSON schema fields and UTF-8 round-trip behavior.
- multiple updates without duplicate records.
- deterministic fallback IDs after the ID strategy is unified.
- malformed persisted data behavior, once the intended policy is decided.

Acceptance criteria:

- All storage and incremental-update behavior runs with a fake embedding backend.

### 12. Add an optional real-model integration test

Improvement:

- Add a separately marked test that loads `bge-small-en-v1.5.Q8_0.gguf` only when an explicit model path is supplied.

Acceptance criteria:

- Default `uv run pytest` remains fast and model-independent.
- The integration test verifies that one input produces a non-empty sequence-level vector.
- Missing model files cause a skip, not a failure.

## Suggested order

1. Stable document IDs.
2. README and intentional-simplification notes.
3. Keyword, cleaning, and PDF tests.
4. Embedding types and injectable backend.
5. PDF resource lifecycle and URI handling.
6. `keep_separator` and token-length semantics when those concepts become the active learning topic.
7. Optional real-model integration test.

## Reference material

- <https://github.com/pguso/rag-from-scratch>
- <https://github.com/pguso/rag-from-scratch/blob/main/examples/03_text_splitting_and_chunking/example.js>
- <https://github.com/pguso/rag-from-scratch/blob/main/examples/04_intro_to_embeddings/02_generate_embeddings/example.js>
- <https://github.com/abetlen/llama-cpp-python>
