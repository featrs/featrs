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
- `cargo clippy --all-targets --all-features -- -W clippy::cognitive_complexity -D warnings` must pass
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

Releases follow the GitHub milestones:

- **Nightly** (patch, `x.y.z`): the `Nightly release` workflow runs daily at
  03:00 UTC. If `main` has code changes since the last tag, it bumps the patch
  version, tags it, publishes to crates.io, and creates a GitHub prerelease.
  Nightlies are regular crates.io versions but are marked as prereleases on
  GitHub.
- **Minor** (`0.x.0`) and **LTS** (`x.0.0`): close the matching GitHub
  milestone, for example `v0.6.0 - Pipeline Composition & Automation` or
  `v1.0.0 - Production Readiness`. The `Milestone release` workflow bumps to
  the milestone version, tags it, publishes to crates.io, and creates the
  GitHub release.
- **Release metadata**: after a milestone release, the workflow opens a
  `chore: sync vX.Y.Z release metadata` PR and dispatches CI on its branch. If
  branch protection permits, it merges automatically; otherwise merge it once
  CI passes. The next milestone release refuses to run until this PR is merged,
  because its changelog would otherwise duplicate the previous release.

Milestone titles must start with a version: `v0.6.0`, `0.6.0`, `v1.0.0`, etc.
Patch milestones (`0.6.1`) are rejected; patch releases are automated
nightlies. To publish a nightly on demand, run the `Nightly release` workflow
manually; it still publishes only when `main` has code changes since the last
tag. To retry a failed milestone release, run the `Milestone release` workflow
manually with the milestone number.

Do not bump `Cargo.toml`, edit a release section by hand, or push `v*` tags
manually. The workflows use the `CARGO_REGISTRY_TOKEN` repository secret, a
crates.io API token with `publish-new` and `publish-update` scopes.
