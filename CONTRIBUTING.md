# Contributing &amp; maintaining

Notes for developing `conlang` and cutting releases. For *using* the package, see the
[README](README.md).

## Development setup

The package is pure Python with no runtime dependencies. Clone it and install in editable mode
with the test extra:

```bash
git clone https://github.com/lordcheetah/Language-Construction-Code.git
cd Language-Construction-Code
python -m pip install -e ".[dev]"
```

## Running the tests

```bash
python -m pytest -q
```

If a stray third-party pytest plugin interferes (some environments autoload a broken one),
disable plugin autoloading:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q     # PowerShell: $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
```

## Building the distributables locally

```bash
python -m pip install --upgrade build
python -m build          # -> dist/conlang-<version>-py3-none-any.whl and .tar.gz
```

## Cutting a release

Releases are automated by [`.github/workflows/release.yml`](.github/workflows/release.yml),
triggered by pushing a **version tag** — a normal branch push does *not* release. To release:

1. Bump `version` in `pyproject.toml` and commit.
2. Tag and push **the tag** (not just the branch):

   ```bash
   git tag v0.2.0 && git push origin v0.2.0
   ```

The workflow then runs the test suite, checks the tag matches the version in `pyproject.toml`,
builds the wheel + sdist, attaches them to a **GitHub Release** (with generated notes), and
publishes them to **PyPI**.

### One-time PyPI setup (trusted publishing)

PyPI upload uses OIDC [trusted publishing](https://docs.pypi.org/trusted-publishers/) — no API
token or stored secret. Configure it once before the first release:

1. On PyPI → *Account settings → Publishing*, add a **pending publisher** for project name
   `conlang`: owner = `lordcheetah`, repository = `Language-Construction-Code`, workflow =
   `release.yml`, environment = `pypi`.
2. In the repo → *Settings → Environments*, create an environment named `pypi`.

If the `pypi-publish` job ever runs before this is set up, it fails while the GitHub Release job
still succeeds; just finish the setup and **re-run the failed job** from the Actions run.

(If the name `conlang` is ever taken on PyPI, rename `project.name` in `pyproject.toml` and the
`url:` in the workflow to match.)
