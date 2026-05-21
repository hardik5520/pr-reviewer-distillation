# REVIEW_RUBRIC

This document defines what counts as a high-quality code review in this
project. It drives three things:

1. The prompt used to generate training reviews via GPT 4o
2. The judge prompt used to evaluate the fine tuned model
3. The reference used when hand labeling the golden eval set

If review style needs to change, update this doc first, then regenerate
the affected data.

## What a good review does

In rough priority order:

1. **Identifies real defects.** Bugs, off by one errors, race conditions,
   broken edge cases, regressions, incorrect logic.
2. **Flags security and safety issues.** Injection risks, credential
   exposure, unsafe deserialization, auth bypasses, missing input
   validation.
3. **Catches missing tests.** New behavior added without tests, edge cases
   not covered, modified logic where existing tests no longer apply.
4. **Notes performance concerns.** O(n^2) on hot paths, unnecessary I/O,
   blocking calls in async contexts, allocation in tight loops.
5. **Suggests maintainability improvements.** Dead code, duplicated logic,
   unclear names that obscure intent, broken abstraction boundaries.
6. **Confirms what is done well, briefly.** One line acknowledgement of
   strong patterns. No generic praise.

Every comment ties to a specific file and line (or hunk). No floating
commentary.

## What a good review does NOT do

- **Nitpick style.** Linters handle trailing commas, line length, import
  order, quote style. Reviews do not.
- **Reword docstrings or comments** unless the original is misleading.
- **Generic praise.** "Looks good!", "Nice work!", "LGTM!" with no
  substance.
- **Bikeshed names** unless the current name is actively misleading.
- **Speculate beyond the diff.** Reviews comment on what was changed,
  not on what was not.
- **Hallucinate problems** that do not exist in the actual code. Better
  to miss a minor issue than invent one.
- **Apply production scrutiny to test files.** Test code has different
  standards: clarity over DRY, explicit over clever.

## Output format

Each review has three parts in this exact order:

### 1. Summary

One to three sentences describing what the change does. Demonstrates
understanding without restating the diff line by line.

### 2. Issues

Zero or more issues, each in this format:

- **`<severity>`** `<file>:<line>` — `<description and suggestion>`

Before concluding "No issues found", explicitly consider each of these
dimensions: test coverage for the new behavior, error handling for the
failure paths, edge cases (empty inputs, large inputs, concurrent access),
naming clarity, and breaking change implications for callers. Only
conclude **No issues found** if all of these have been considered and
none apply. Do not fabricate issues to fill space, but do not default
to passive either.

### 3. Strengths

Optional, one line max. Skip unless the change shows an unusually good
pattern worth calling out.

## Severity definitions

- **critical**: Bug, security issue, data loss risk, regression that
  could break production. Must be addressed before merge.
- **major**: Significant design flaw, missing test for new behavior,
  performance regression on a hot path. Should be discussed before merge.
- **minor**: Readability, naming, small refactor opportunity. Author
  may address or defer.

## Tone

Direct, technical, no hedging. Do not start with "I think" or "perhaps."
State the issue and the fix. Reviewers serve the code, not the author's
feelings.

## Scoring dimensions (for eval only)

When grading a generated review against a golden reference, score on:

- **Precision**: of issues flagged, fraction that are real and worth
  raising
- **Recall**: of real issues in the diff, fraction the review caught
- **Specificity**: are issues tied to specific lines, or vague?
- **Actionability**: can the author act on each comment without
  asking for clarification?
- **Brevity**: concise, or padded with filler?

A review that catches the one critical bug and stops there beats a
review that lists ten nits and misses the bug.

## Example: same diff, bad review vs good review

**Diff**: `get_user_preferences` adds an in memory cache, but the
companion `update_user_preferences` function does not invalidate it.

**Bad review**:

> Looks good overall. Nice use of caching for performance. Maybe add
> a comment explaining why caching is needed. Also consider renaming
> `data` to `user_data` for clarity.

**Good review**:

> ### Summary
> Adds an in memory cache to `get_user_preferences` keyed by user_id.
>
> ### Issues
> - **major** `src/users/preferences.py:42` — Cache is populated on
>   read but never invalidated when the user record changes.
>   `update_user_preferences` at line 78 should call
>   `cache.pop(user_id)`, otherwise the next read returns stale data.
> - **minor** `src/users/preferences.py:45` — No cache size bound.
>   Long running processes will leak memory. Use
>   `functools.lru_cache(maxsize=1024)` or an explicit LRU.
>
> ### Strengths
> Test in `tests/test_preferences.py` covers the cache hit path.

The bad review reads like a LinkedIn comment. The good review changes
how the code ships.