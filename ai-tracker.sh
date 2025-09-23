#!/bin/bash

# A script to track changes, update UPDATES.md, and commit.

set -e

# Check if inside a git repository
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1;
then
    echo "Error: Not in a git repository."
    exit 1
fi

# Get the root directory of the git repository
GIT_ROOT=$(git rev-parse --show-toplevel)
UPDATES_FILE="$GIT_ROOT/UPDATES.md"

# Arguments
REF=${2:-HEAD}

# Stage all changes
git add -u

# Determine diff command based on arguments
if [ "$1" == "update" ]; then
    DIFF_COMMAND="git diff --staged"
    if [ "$REF" != "HEAD" ]; then
        DIFF_COMMAND="git diff $REF..HEAD"
    fi
else
    echo "Usage: ai-tracker update [--ref <commit-hash> | <branch-name>]"
    exit 1
fi

# Get the diff output, filtering for non-code changes
DIFF=$(eval "$DIFF_COMMAND" | grep -E '^(\+\+\+|---|\+|\-|@@)' | grep -v '^[+ ]*$' | grep -v '^-*$')

if [ -z "$DIFF" ]; then
    echo "No relevant changes found. Exiting."
    exit 0
fi

# Use a temporary file to store Gemini's output
TEMP_MD=$(mktemp)

# Call Gemini to summarize the changes
gemini --prompt="
### ROLE & PERSONA ###
You are an expert technical writer.

### CONTEXT ###
The user has provided a git diff output. Your task is to analyze the changes and generate a summary suitable for a project update log.

### PRIMARY TASK ###
Summarize and categorize the changes from the git diff. Use a category for each change (e.g., 'What's New', 'Bugfix', 'Refactor'). The output must be valid markdown following the exact structure shown in the example.

### SPECIFICATIONS & INSTRUCTIONS ###
- Group changes by category, then by file.
- Do not include spaces, newlines, or whitespace-only changes.
- Do not include any conversational text outside the formatted summary.
- The most recent update must be placed at the top of the file.

### EXAMPLE FORMAT ###
## $(date '+%Y-%m-%d')

### What's New

#### \`src/features/auth.js\`

- Implemented the core logic for the new user authentication flow.
- Added a new component for the login form.

### Bugfix

#### \`src/components/Header.js\`

- Fixed a layout issue where the logo would overlap navigation links on smaller screens.

### OUTPUT FORMAT & CONSTRAINTS ###
- Provide your response exclusively as the raw text of the summary.
- DO NOT include any explanations or introductory sentences.
- The current date is $(date '+%Y-%m-%d').

Act autonomously. Do not ask for clarification. Begin analysis immediately when invoked.

${DIFF}
" > "$TEMP_MD"

# Prepend the new content to UPDATES.md
if [ -f "$UPDATES_FILE" ]; then
    CURRENT_CONTENT=$(cat "$UPDATES_FILE")
    echo -e "$(cat \"$TEMP_MD\")\n\n${CURRENT_CONTENT}" > "$UPDATES_FILE"
else
    cat "$TEMP_MD" > "$UPDATES_FILE"
fi

# Clean up temp file
rm "$TEMP_MD"

# Add the updated UPDATES.md to staged files
git add "$UPDATES_FILE"

# Generate commit message and commit
COMMIT_MSG=$(gemini --prompt="
### ROLE & PERSONA ###
You are an expert at generating git commit messages.

### CONTEXT ###
The user has provided a summary of code changes.

### PRIMARY TASK ###
Based on the summary, generate a concise and relevant git commit message.

### SPECIFICATIONS & INSTRUCTIONS ###
- The first line should be the subject, following Conventional Commits format (e.g., 'feat: add new feature').
- The subject should be a maximum of 50 characters.
- The body should contain a brief, 1-2 sentence description if necessary.

### EXAMPLE FORMAT ###
feat: add user authentication
This commit adds a new user authentication flow using OAuth 2.0.

### OUTPUT FORMAT & CONSTRAINTS ###
- Provide your response as the raw commit message.
- Do not include any explanations or conversational text.

Act autonomously. Do not ask for clarification. Begin analysis immediately when invoked.

${DIFF}
")

git commit -m "$COMMIT_MSG"

echo "Changes summarized, UPDATES.md updated, and committed."