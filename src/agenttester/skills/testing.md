## Testing Process

After making changes, validate your work by running the existing test suite. Check for test commands in this order: CI workflow files (`.github/workflows/`, `.gitlab-ci.yml`, `Jenkinsfile`), Makefile targets (`make test`), pyproject.toml scripts, package.json scripts, or common test runners (`pytest`, `npm test`, `go test ./...`).

CI workflow files are the most reliable source of truth for how the project is tested and linted — they reflect exactly what must pass before code is merged. If a workflow exists, read it before running tests so you use the same commands, flags, and environment the CI pipeline uses.

In repositories that have linting configured, also run the linter and auto-fix any issues that will not break the code.

Do not consider a task complete until tests pass.

## Writing new tests

When you add a new function, class, configuration option, or code path, add tests for it. Do not leave new code untested.

Before writing a test, look at the existing test files for the module you changed to understand the conventions: how tests are structured, what helpers exist, and what level of coverage is expected. Match that style.

A test should verify the observable behaviour of the code, not its implementation details. Test what the function does, not how it does it. Avoid testing private internals directly when the same behaviour is reachable through the public interface.

Each test should cover one thing. If you need to verify multiple behaviours of a new feature, write multiple tests rather than one large test that checks everything.
