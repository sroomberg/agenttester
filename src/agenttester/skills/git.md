## Git Operations

You may perform the following git operations without asking for permission:

- **Stage and commit**: `git add <files>` then `git commit -m "<message>"`
- **Push a branch**: `git push -u origin <branch-name>` (only the branch you are working on)
- **Pull**: `git pull` or `git fetch && git merge origin/<branch>`
- **Rebase**: `git rebase origin/main` (or the default branch) to update your branch
- **Update from the default branch**: `git fetch origin && git rebase origin/main` (preferred over merge for a clean history)

### Your branch

You are working on a pre-created branch in the format:

```
agenttester/<model-name>/<session-name>
```

Check your current branch with `git branch --show-current`. **Do not create new branches** — commit all changes to your assigned branch.

### Rules

- **Never push to the default branch** (main, master, or trunk) directly.
- **Never force-push** unless explicitly instructed.
- **Never amend published commits** (commits already pushed to a remote branch).
- **Never create new branches** — your branch is pre-assigned; use it for all commits.
- Before committing, verify that tests pass and the linter is clean.
- Write concise, descriptive commit messages in the imperative mood ("add feature", "fix bug", not "added" or "fixes").
- If a rebase produces conflicts you cannot resolve safely, abort with `git rebase --abort` and report the situation.
