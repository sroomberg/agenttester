## Code Review Standards

When reviewing or evaluating code — your own or another agent's — apply the following standards. Flag any violation clearly and explain why it matters.

### Language and Dependencies

- **Do not introduce a new language or runtime** unless the task is provably impossible without it. If the codebase is Python, the solution is Python. Adding a second language (shell scripts that could be Python, Node scripts in a Python repo, etc.) creates a maintenance burden and splits expertise.
- **Do not add a new dependency** unless the standard library or existing dependencies cannot do the job. Every new package is a version conflict waiting to happen.
- **Do not add configuration formats or build tools** that aren't already in use. One config system, one build tool.

### Single Source of Truth

- **Each piece of logic or data must live in exactly one place.** If the same value, rule, or algorithm appears in two files, one of them will eventually be wrong.
- Watch for duplicated constants, copy-pasted validation logic, parallel data structures that represent the same concept, and documentation that restates what the code already says.
- When you find duplication, consolidate it — extract a shared function, constant, or module — rather than adding a third copy.

### Code Reuse

- **Search before you write.** Before implementing any utility, query the codebase for existing functions that do the same or similar thing. Use existing helpers even if they require a small generalisation; do not write a second implementation that diverges over time.
- Prefer extending an existing abstraction over creating a parallel one. Two abstractions that serve the same role will drift apart.

### Code Smells to Flag

- **Long functions**: any function that cannot be understood at a glance. If it does more than one thing, split it.
- **Deep nesting**: more than two or three levels of indented control flow is a sign the logic should be restructured or extracted.
- **Magic values**: unnamed literals (`42`, `"admin"`, `"/api/v1"`) embedded in logic. Extract them as named constants.
- **Dead code**: commented-out blocks, unused imports, unreachable branches, and functions with no callers.
- **Inconsistent naming**: variables or functions that break the surrounding conventions, or names that are misleading about what the thing actually does.
- **Inappropriate intimacy**: code that reaches into the internals of another module when a public interface exists or should exist.
- **Speculative generality**: abstractions, parameters, or extension points added for hypothetical future requirements that don't exist yet.
- **God objects**: classes or modules that know about and control too much. Responsibility should be distributed.

### What Good Code Looks Like

- A reader unfamiliar with the codebase can understand a function's purpose from its name and signature alone.
- Changes are localised — modifying behaviour requires editing one place, not hunting for all the copies.
- The solution is as simple as the problem allows. Complexity is not added in anticipation of requirements that haven't been asked for.
