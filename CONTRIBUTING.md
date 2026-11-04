# Contributing

## Getting started

```bash
git clone https://github.com/featrs/featrs.git
cd featrs
cargo build
cargo test
```

## Code style

- `cargo fmt` before committing
- `cargo clippy --all-targets --all-features -- -D warnings` must pass
- 4-space indentation, no trailing whitespace
- All public items must have doc comments (`///`)
- Module-level `//!` docs for every `.rs` file

## Adding a new transformer

1. Create a struct with `fitted: bool` and parameter fields
2. Implement `Fit<DataFrame, DataFrame>` and `Transform<DataFrame>`
3. Add module docs (`//!`), struct docs (`///`), and constructor docs
4. Add tests in a `#[cfg(test)] mod tests` block
5. Register the module in the parent `mod.rs`

## PR process

1. Branch from `main`: `git checkout -b feat/my-feature`
2. Commit with conventional commits style (`feat:`, `fix:`, `docs:`, etc.)
3. Push and create a PR
4. CI must pass (build, clippy, fmt, test)
5. Squash-merge when approved

## Releases

Releases are automated with [release-plz](https://release-plz.dev):

1. Every push to `main` opens or updates a release PR. It bumps the version in
   `Cargo.toml` and adds a `CHANGELOG.md` entry from the conventional commits.
2. Review the release PR. Edit the changelog entry if it needs more detail.
3. Merge the release PR. The `Release` workflow then publishes the crate to
   crates.io, pushes the `vX.Y.Z` tag, and creates the GitHub release.

Do not bump the version or push release tags by hand.

Repository secrets used by the workflow:

- **`CARGO_REGISTRY_TOKEN`** (configured): a crates.io API token with
  `publish-new` and `publish-update` scopes.
- **`RELEASE_PLZ_TOKEN`** (recommended): a fine-grained PAT with
  `contents: write` and `pull-requests: write` on this repository. Without it
  the workflow falls back to `GITHUB_TOKEN`, but CI will not run on the release
  PR because GitHub suppresses workflow runs for `GITHUB_TOKEN`-created PRs.

To switch to crates.io trusted publishing later, remove the
`CARGO_REGISTRY_TOKEN` environment variable from `.github/workflows/release.yml`,
add `id-token: write` to the `release` job permissions, and configure the
trusted publisher on crates.io.
