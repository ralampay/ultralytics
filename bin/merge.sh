#!/usr/bin/env bash
set -euo pipefail

upstream_remote="${UPSTREAM_REMOTE:-upstream}"
fork_remote="${FORK_REMOTE:-origin}"
base_branch="${BASE_BRANCH:-main}"
commit_message="${COMMIT_MESSAGE:-bump}"
branch="$(git branch --show-current)"

if [[ -z "$branch" ]]; then
  echo "error: must be run from a local branch" >&2
  exit 1
fi

if [[ "$branch" == "$base_branch" ]]; then
  echo "error: refusing to update '$base_branch' directly; run this from a feature or sync branch" >&2
  exit 1
fi

for remote in "$fork_remote" "$upstream_remote"; do
  if ! git remote get-url "$remote" >/dev/null 2>&1; then
    echo "error: git remote '$remote' is not configured" >&2
    exit 1
  fi
done

if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
  echo "Committing local changes on $branch..."
  git add -A
  git commit -m "$commit_message"
fi

echo "Fetching $fork_remote and $upstream_remote..."
git fetch --prune "$fork_remote"
git fetch --prune --tags "$upstream_remote"

upstream_branch="$upstream_remote/$base_branch"
if ! git show-ref --verify --quiet "refs/remotes/$upstream_branch"; then
  echo "error: remote branch '$upstream_branch' does not exist" >&2
  exit 1
fi

echo "Merging $upstream_branch into $branch..."
git merge --no-edit -m "$commit_message" "$upstream_branch"

if ! git merge-base --is-ancestor "$upstream_branch" HEAD; then
  echo "error: HEAD does not contain $upstream_branch after merge" >&2
  exit 1
fi

echo "Pushing $branch to $fork_remote..."
git push -u "$fork_remote" "$branch"

fork_url="$(git remote get-url "$fork_remote")"
fork_path="${fork_url#git@github.com:}"
fork_path="${fork_path#https://github.com/}"
fork_path="${fork_path%.git}"

echo "Sync branch pushed successfully."
if [[ "$fork_path" != "$fork_url" ]]; then
  echo "Open a pull request: https://github.com/$fork_path/compare/$base_branch...$branch?expand=1"
fi
