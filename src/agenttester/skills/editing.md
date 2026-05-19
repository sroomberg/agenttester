You are allowed to read and edit any file in the repository without asking for permission first.

## Before Adding New Code

Search the codebase to build context before implementing anything new. The functionality you need may already exist or be partially implemented elsewhere. Use `grep`, `find`, or similar tools to look for related functions, utilities, or patterns before writing new ones.

When writing or editing a function, consider whether it belongs where it currently lives or whether it would be better placed in a shared utility module so it can be referenced from a single location. If you find yourself duplicating logic across multiple files, consolidate it.

## Single Source of Truth

Each piece of logic or data must live in exactly one place. If the same value, rule, or algorithm appears in two files, one of them will eventually be wrong. When you find duplication, consolidate it — extract a shared function, constant, or module — rather than adding a third copy.

## Readability

Prioritize readability when writing code. Prefer clear, descriptive names over brevity. Keep functions small and focused on a single responsibility. Structure code so a reader can follow the logic without needing to trace through multiple layers of indirection. Straightforward code that is easy to read and reason about is better than clever code that is hard to follow.

## Completing Work

When you finish a task or a set of changes, always produce a concise summary of what you changed: which files were modified, what was added or removed, and why. This gives the user a quick way to verify the work without reading every diff.

## Things to Avoid

- **Magic values**: embed no unnamed literals (`42`, `"admin"`, `"/api/v1"`) in logic — extract them as named constants.
- **Deep nesting**: more than two or three levels of indented control flow is a sign the logic should be restructured or extracted.
- **Dead code**: do not leave commented-out blocks, unused imports, or unreachable branches.
- **Inappropriate intimacy**: do not reach into another module's internals when a public interface exists or should exist.
- **Speculative generality**: do not add abstractions, parameters, or extension points for hypothetical future requirements.
- **God objects**: do not let a single class or module accumulate unrelated responsibilities.
