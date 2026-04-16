# Contributing to SED_Tools

Thank you for your interest in contributing to SED_Tools.

## Reporting Bugs & Requesting Features

Open an issue on GitHub. For bugs, include:

- A clear description of the problem
- Steps to reproduce it
- What you expected vs. what happened
- Relevant environment details (OS, Python version, relevant file sizes/grid names if applicable)

For feature requests, describe the use case and why it would be useful.

## Contribution Workflow

1. Fork the repository
2. Create a branch for your change
3. Make your changes
4. Open a pull request against `main` with a clear description of what you changed and why

Keep PRs focused — one logical change per PR. Large or sweeping changes should be discussed in an issue first.

## Code Style

No strict requirements. Match the style of the surrounding code you are editing.

## What to Contribute

Areas where contributions are most useful:

- Bug fixes with a clear root cause
- New stellar model grid support
- Performance improvements to processing or I/O (especially at large data scales)
- Documentation corrections

## What Not to Do

- Do not open PRs that add unrelated changes bundled together
- Do not submit changes that have not been tested against real data
- Do not modify `spectra_cleaner.py` unit standardization logic without opening an issue and discussing it first — it is the single authority for unit conversion across the entire pipeline

## Contact

For questions not suited to a public issue: **niall.j.miller@gmail.com**
