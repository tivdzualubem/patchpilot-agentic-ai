"""Compatibility entry point for the canonical research summarizer.

This script replaces the obsolete 20-task, three-condition Mutmut-only
summarizer while preserving the historical command name.
"""

if __name__ == "__main__":
    __import__("runpy").run_path(
        __import__("pathlib")
        .Path(__file__)
        .with_name("summarize_research_evaluation.py"),
        run_name="__main__",
    )
