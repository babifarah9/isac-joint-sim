#!/usr/bin/env bash
set -euo pipefail

# One-time publishing helper. Requires git and the authenticated GitHub CLI.
REPO="${1:-isac-joint-sim}"
if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI missing. Install it (macOS): brew install gh"
  exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
  echo "Authenticate first: gh auth login"
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "Git missing. Install git then retry."
  exit 1
fi
cd "$(dirname "$0")"
if [[ ! -d .git ]]; then
  git init -b main
fi
git add app.py src tests docs README.md LICENSE pyproject.toml requirements.txt .gitignore publish_to_github.sh
if ! git diff --cached --quiet; then
  git commit -m "Initial ISAC joint sensing and communications simulator"
fi
if git remote get-url origin >/dev/null 2>&1; then
  git push -u origin main
else
  # The GitHub CLI uses the authenticated account; never asks for password in script.
  gh repo create "$REPO" --public --source=. --remote=origin --push
fi
echo "Repository published. Check: $(git remote get-url origin)"
