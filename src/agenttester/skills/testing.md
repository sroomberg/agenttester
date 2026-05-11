After making changes, validate your work by running the existing test suite. Check for test commands in this order: Makefile targets (`make test`), pyproject.toml scripts, package.json scripts, or common test runners (`pytest`, `npm test`, `go test ./...`).

In repositories that have linting configured, also run the linter and auto-fix any issues that will not break the code.

Do not consider a task complete until tests pass.