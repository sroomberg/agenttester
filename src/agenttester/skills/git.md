## Git Operations

You may perform the following git operations without asking for permission:

- **Stage and commit**: `git add <files>` then `git commit -m "<message>"`
- **Push a branch**: `git push -u origin <branch-name>` (only the branch you are working on)
- **Pull**: `git pull` or `git fetch && git merge origin/<branch>`
- **Rebase**: `git rebase origin/main` (or the default branch) to update your branch
- **Update from the default branch**: `git fetch origin && git rebase origin/main` (preferred over merge for a clean history)

### Your branch

A branch is created automatically when you first write a file. Its name is derived from the task prompt:

```
agenttester/<model-name>/<prompt-slug>
```

Check your current branch with `git branch --show-current`. **Do not create new branches** — commit all changes to your assigned branch.

### Merge conflicts

Resolve conflicts automatically without asking for confirmation:

1. Run `git status` to identify conflicted files.
2. Read each conflicted file and resolve by applying the correct combination of both sides.
3. Stage resolved files with `git add <file>`.
4. Continue the operation (`git rebase --continue` or `git merge --continue`).
5. If a conflict cannot be resolved safely (e.g., incompatible semantic changes), abort with `git rebase --abort` or `git merge --abort` and report the situation.

### Rules

- **Never push to a repository other than the one you are working in.** You may only push to remotes that were configured in the working repository when you started. Do not add new remotes and do not use the `bash` tool to push to external URLs.
- **Never push to the default branch** (main, master, or trunk) directly.
- **Never force-push** unless explicitly instructed.
- **Never amend published commits** (commits already pushed to a remote branch).
- **Never create new branches** — your branch is assigned automatically; use it for all commits.
- Before committing, verify that tests pass and the linter is clean.
- Write concise, descriptive commit messages in the imperative mood ("add feature", "fix bug", not "added" or "fixes").
