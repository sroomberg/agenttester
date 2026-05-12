You are allowed to read and edit any file in the repository without asking for permission first.

## Before Adding New Code

Search the codebase before implementing anything new. The functionality you need may already exist or be partially implemented elsewhere. Use `grep`, `find`, or similar tools to look for related functions, utilities, or patterns before writing new ones.

When writing or editing a function, consider whether it belongs where it currently lives or whether it would be better placed in a shared utility module so it can be referenced from a single location. If you find yourself duplicating logic across multiple files, consolidate it.

## Readability

Prioritize readability when writing code. Prefer clear, descriptive names over brevity. Keep functions small and focused on a single responsibility. Structure code so a reader can follow the logic without needing to trace through multiple layers of indirection. Straightforward code that is easy to read and reason about is better than clever code that is hard to follow.
