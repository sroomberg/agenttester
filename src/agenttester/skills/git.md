## Git Operations

You may perform the following git operations without asking for permission:

- **Create a branch**: `git checkout -b <branch-name>` or `git switch -c <branch-name>`
- **Stage and commit**: `git add <files>` then `git commit -m "<message>"`
- **Push a branch**: `git push -u origin <branch-name>` (only the branch you are working on)
- **Pull**: `git pull` or `git fetch && git merge origin/<branch>`
- **Rebase**: `git rebase origin/main` (or the default branch) to update your branch
- **Merge**: `git merge <branch>` to bring changes into your current branch
- **Update from the default branch**: `git fetch origin && git rebase origin/main` (preferred over merge for a clean history)

### Rules

- **Never push to the default branch** (main, master, or trunk) directly. Always push to a feature branch.
- **Never force-push** unless explicitly instructed.
- **Never amend published commits** (commits already pushed to a remote branch).
- Before committing, verify that tests pass and the linter is clean.
- Write concise, descriptive commit messages in the imperative mood ("add feature", "fix bug", not "added" or "fixes").
- If a rebase produces conflicts you cannot resolve safely, abort with `git rebase --abort` and report the situation.
