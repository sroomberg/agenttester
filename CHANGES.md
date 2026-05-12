# Changelog

## v0.5.1 — 2026-05-12
- Consolidate duplicated patterns across `agent_runner`, `repl`, and `skills`
- Consolidate config path resolution into `get_config_paths()`; fix REPL global config fallback
- Fix CI publish workflow: move `id-token: write` to workflow level for trusted publishing

## v0.5.0 — 2026-05-12
- Add interactive agent input routing and idle pause/resume

## v0.4.5 — 2026-05-12
- Include `skills.py` priority logic and lock file

## v0.4.4 — 2026-05-12
- Add git/bash built-in skills; prioritise user skills over built-ins

## v0.4.3 — 2026-05-12
- Remove agent count cap

## v0.4.2 — 2026-05-12
- Store reports in global config dir by default
- Support `.yml` and `.yaml` config file extensions

## v0.4.1 — 2026-05-12
- Update packages for wheel compatibility

## v0.4.0 — 2026-05-12
- Consolidate global config to `~/.config/agenttester/config.yml`

## v0.3.7 — 2026-05-11
- Add skills directory system and fix test path resolution

## v0.3.6 — 2026-05-11
- Add connection check on REPL startup with `--skip-checks` / `-S` flag

## v0.3.5 — 2026-05-11
- Add spinner while querying models in REPL

## v0.3.4 — 2026-05-11
- Open REPL by default when run with no subcommand

## v0.3.3 — 2026-05-11
- Fix `agent-tester` binary name in REPL, config example, and tests

## v0.3.2 — 2026-05-11
- Standardize on `agent-tester` binary and config filename

## v0.3.1 — 2026-05-11
- Read version from package metadata
- Replace `at` alias with `agent-tester` to avoid conflict with Unix job scheduler

## v0.3.0 — 2026-05-11
- Add branch name injection, auto-pull, and global config support

## v0.2.0 — 2026-05-11
- Add cost tracking system with local storage and CLI interface

## v0.1.1 — 2026-05-08
- Add README

## v0.1.0 — 2026-05-08
- Initial release
