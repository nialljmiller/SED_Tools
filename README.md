# SED Tools

## Data locations

The tools and Python APIs now write their outputs to `./data` at the repository
root by default (for example `./data/stellar_models`). To change where data is
stored, set one of the environment variables before running a command:

- `SED_DATA_DIR` to change the base `./data` directory for both filters and
  stellar models at once.
- `SED_STELLAR_DIR` or `SED_FILTER_DIR` to override either location
  individually.

The resolved paths are also exposed programmatically as
`sed_tools.DATA_DIR_DEFAULT`, `sed_tools.STELLAR_DIR_DEFAULT`, and
`sed_tools.FILTER_DIR_DEFAULT` to make them easy to inspect or override in
code.
