# firedrake-ts &middot; [![ci-firedrake-ts](https://github.com/Atomic-Industries/firedrake-ts/actions/workflows/ci-firedrake-ts.yaml/badge.svg)](https://github.com/Atomic-Industries/firedrake-ts/actions/workflows/ci-firedrake-ts.yaml)

The firedrake-ts library provides an interface to PETSc TS for the scalable solution of DAEs arising from the discretization of time-dependent PDEs.

This is Atomic Industries' fork of [IvanYashchuk/firedrake-ts](https://github.com/IvanYashchuk/firedrake-ts), which has been inactive for several years.

## Example

Check `examples/` for the examples.

## Requirements

Python 3.12 or newer and a working [Firedrake](https://firedrakeproject.org/install.html) installation.

Firedrake is **not** declared as a dependency of this package: it is expected to come from the environment, in practice the `firedrakeproject/firedrake-vanilla-default` image. Declaring it would make every `uv sync` of a downstream project try to resolve and build Firedrake and PETSc from source.

## Installation

Released versions are published to the [Atomic private PyPI registry](https://pypi.atmc.dev), which requires a connection to our [Tailscale VPN](https://tailscale.com).

### Using the UV project manager

```bash
uv add firedrake-ts --index https://pypi.atmc.dev
```

### Using good ol' PIP

```bash
pip install firedrake-ts --extra-index-url https://pypi.atmc.dev
```

### Straight from the repository

```bash
uv add "firedrake-ts @ git+https://github.com/Atomic-Industries/firedrake-ts.git@master"
```

## Development

Work inside a container built on the Firedrake image, so that Firedrake and PETSc come from the image rather than from a local build. Firedrake lives in the image's system Python, so the project environment has to inherit it:

```bash
uv venv --system-site-packages
uv sync --inexact --group dev
```

`--inexact` keeps the pytest plugins that ship with the image from being pruned. Then run the tests:

```bash
uv run pytest
```

Lint and format with [ruff](https://docs.astral.sh/ruff/):

```bash
uv run ruff check .
uv run ruff format .
```

Optionally install the pre-commit hooks, which run the same two commands on staged files:

```bash
uvx pre-commit install
```

### Release

Bump `version` in `pyproject.toml` and push a matching `vX.Y.Z` tag on `master`. CI lints, tests and creates the GitHub release with the built sdist and wheel attached; it fails the release if the tag and the version disagree. Bugfixes bump the patch version, new features the minor version, and backward incompatible changes the major version.

Then upload to the Atomic registry from a machine on the VPN:

```bash
uv build && uv publish --index=atomic --username="" --password=""
```

## Reporting bugs

If you found a bug, create an [issue].

[issue]: https://github.com/Atomic-Industries/firedrake-ts/issues/new

## Contributing

Pull requests are welcome. Clone the repository:

```bash
git clone https://github.com/Atomic-Industries/firedrake-ts.git
```

Make your change and add tests for it. Make the tests pass, and check that `ruff check .` and `ruff format --check .` are clean. Then [submit a pull request][pr].

[pr]: https://github.com/Atomic-Industries/firedrake-ts/pulls
