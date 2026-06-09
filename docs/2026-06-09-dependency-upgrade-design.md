# candle-video Dependency Upgrade Design

## Goal

Update `candle-video` direct dependencies to current stable releases, regenerate the dependency lock state, and restore a working build with minimal compatibility fixes.

## Scope

- Update versions in `Cargo.toml` for all direct dependencies and dev-dependencies.
- Refresh lock resolution after manifest changes.
- Run build verification and fix code that breaks due to dependency API changes.
- Keep fixes tightly scoped to compatibility issues caused by the upgrade.

## Non-Goals

- No unrelated refactoring.
- No feature work.
- No broad test rewrites unless required by dependency breakage.

## Recommended Execution Strategy

1. Determine current latest stable versions for all direct dependencies.
2. Update `Cargo.toml` in one pass.
3. Regenerate dependency resolution and inspect breakages.
4. Apply minimal source changes needed for compatibility.
5. Re-run build and targeted test verification until green.

## Risks

- The Candle stack may require synchronized version bumps across `candle-core`, `candle-nn`, `candle-transformers`, and `candle-flash-attn`.
- `tokenizers`, `hf-hub`, `clap`, and tracing-related crates may introduce API changes.
- Some failures may come from transitive updates rather than direct version edits.

## Verification

- `cargo check`
- `cargo test --no-run` if feasible
- More targeted commands if a full test compile is too expensive
