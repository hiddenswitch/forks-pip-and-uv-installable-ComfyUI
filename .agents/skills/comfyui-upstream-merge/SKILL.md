---
name: comfyui-upstream-merge
description: >-
  Use this skill when merging upstream ComfyUI (Comfy-Org/ComfyUI master or a release tag) into this
  fork's develop branch, resolving the conflicts that merge produces, or cutting a fork release
  (vA.B.C.N) from develop afterwards.
---

# Skill: comfyui-upstream-merge

The compatibility rules for upstream code live in [docs/merging.md](../../../docs/merging.md);
read it first and apply it to every incoming file. This skill adds the procedure around it and the
mistakes already made. Study prior merges with `git log --merges --oneline origin/develop | grep -i upstream`
and read their bodies; good examples are `c21040627`, `3e6191ffd`, `cf72794f5`, `44af48d70`.

## Merge procedure

1. `git fetch comfyui master` (the `comfyui` and `upstream` remotes use `tagOpt --no-tags`: upstream
   tags such as `v0.9.2` collide with fork tags of the same name). Work in a worktree
   `~/Documents/worktrees-comfy-upstream-<date>` on `merge/upstream-<date>` from `origin/develop`.
2. `git merge --no-ff comfyui/master`. Never resolve a whole directory with "ours". Upstream renames
   that need moving (`comfy_extras/nodes_*.py` → `comfy_extras/nodes/`, `tests-unit/` →
   `tests/unit/`, root modules → `comfy/cmd/`) keep the fork's location and package-relative imports.
3. `requirements.txt` stays deleted. Move upstream pin bumps into `pyproject.toml`, then
   `ci/generate-locks.sh --upgrade-package <pkg> ...` and check that only those package families moved
   (diff torch and triton across all seven locks; `--upgrade` alone moves them).
4. When upstream removes a feature, remove it from the fork's own surfaces too (CLI/typer options,
   `Configuration`, `MODEL_MANAGEMENT_ARGS`, docs, tests): DirectML in `ae5b8ea8d`.
5. Apply `docs/merging.md`: translate `PromptServer.instance.send_*` to
   `current_execution_context().server` (`52baae9c7`).
6. Delete upstream CI workflows that conflict with the fork's conventions, e.g.
   `.github/workflows/check-ai-co-authors.yml` (`cbc54c915`): fork commits carry agent co-author
   trailers.
7. Check: `ruff check comfy/ comfy_extras/ comfy_api/ comfy_api_nodes/ comfy_compatibility/ comfy_execution/`,
   `pylint --disable=W4902 <paths>` (custom checkers in `tests/*_checker.py`, see `docs/linting.md`),
   the affected unit tests, imports of every changed node module, and no new duplicate `node_id`s
   compared with develop.
8. Commit with a body that states the non-textual carry-overs; open a PR to develop.

## Release procedure (vA.B.C.N, A.B.C = tracked upstream version)

1. Bump `version` in `pyproject.toml` and `__version__` in `comfy/__init__.py` together (checker W9101).
2. Push to develop and wait for Backend Tests, Python Validation, Unit coverage and the CUDA and ROCm
   image workflows to pass on that SHA.
3. **Fast-forward** master to that exact SHA (`git push origin <sha>:refs/heads/master`), tag the same
   SHA (`git tag -a vX -m "ComfyUI X" <sha>`), push the tag, `gh release create vX --latest`.
4. Verify `build-wheel.yml` attached the wheel and sdist and the master/tag publish jobs retagged the
   images.

## Mistakes to avoid

- **`merge --no-ff` for a release**: `docker-build.yml` only retags `<sha>-cuda` images that develop
  validated, so a new merge commit on master cannot publish (v0.45.0.1 at `0e346cd39`; recovered by
  fast-forwarding develop to it, waiting for develop's image builds, then `gh run rerun --failed`).
- **Pushing to develop while a release candidate's CI runs**: develop workflows use
  `cancel-in-progress`, so the new push cancels the candidate's runs. Merge PRs into develop only
  after the candidate has passed and been tagged.
- **Wrong version after a merge**: `63c7a7608` bumped 0.43.0.5 after merging 0.45; fixed in `002addce2`.
- **Absolute imports from upstream code**: `848f31544`, `bd8866861`, `66b1503f6`.
- **Pins dropped by version resolution**: `0ce10c1cd`, `39d7d541b`; torch bumped by lock regeneration: `9918f9eda`.
- **Import shadowing introduced while resolving**: `819eeb192`.
- **Startup side effects lost without a textual conflict** (DynamicVRAM): `cfdb6f513`.
- **Chained `git merge -F -`** silently failed and the tag landed on the old master (v0.35.0.1).
- **Intermittent CI failures are fixed, not rerun**: the no-assets server test had a 30 s startup
  budget against a 25 s cold start (`b614c9c7c`).
