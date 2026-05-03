#!/usr/bin/env bash
# Copy the deployable files from the source project into the Space repo.
#
# Usage (from the empty Space repo directory you cloned from huggingface.co):
#   bash <PATH_TO_PROJECT>/scripts/copy_to_space.sh <PATH_TO_PROJECT>
#
# Example:
#   bash "C:/Users/THUNDER/Desktop/Projects/Video RAG/scripts/copy_to_space.sh" \
#        "C:/Users/THUNDER/Desktop/Projects/Video RAG"

set -e

SRC="${1:?Usage: copy_to_space.sh <project-root>}"

if [ ! -f "$SRC/app.py" ]; then
    echo "ERROR: $SRC does not look like the Video RAG project root (no app.py)" >&2
    exit 1
fi
if [ ! -f "$SRC/chroma_db.tar.gz" ]; then
    echo "ERROR: $SRC/chroma_db.tar.gz is missing." >&2
    echo "Run from the project: python -m scripts.bundle_chroma_db" >&2
    exit 1
fi

echo "Copying deployable files from $SRC -> $(pwd)..."

cp "$SRC/app.py"                   .
cp "$SRC/requirements.txt"         .
cp "$SRC/chroma_db.tar.gz"         .
cp "$SRC/SPACE_README.md"          README.md      # Spaces reads frontmatter from README.md
cp -r "$SRC/src"                   .
cp -r "$SRC/scripts"               .

# Don't ship the local .env or the raw chroma_db dir — secrets/large.
rm -f .env
rm -rf chroma_db

echo
echo "Done. Files in this Space repo:"
ls -la
echo
echo "Next:"
echo "  git add ."
echo "  git commit -m 'Initial deploy'"
echo "  git push"
