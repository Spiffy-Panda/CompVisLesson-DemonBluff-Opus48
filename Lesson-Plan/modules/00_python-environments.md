# Module 00 — Reproducible Python environments: venv, virtualenv, pip, conda, uv

**The problem (in the pipeline):** A CV course is only as good as the instructions a learner can actually follow. If "pip install everything and hope" is the setup story, learners on different machines get different package versions, different binary wheel builds, and subtly different behaviour. Dependency drift is one of the leading reasons course exercises stop working six months after they are written. Before a single line of CV code runs, the pipeline needs a reproducible environment that every learner can reconstruct — and the course needs to explain *why* the approach was chosen over its fast-growing alternatives.

**What you'll be able to do:**

1. Explain the distinction between a Python *interpreter*, a *virtual environment*, and a *package installer* — and why all three are separate concerns.
2. Describe where `venv`, `virtualenv`, `pip`, `conda`, and `uv` each sit in that picture, what problem they each solve, and where they break down.
3. Set up the project environment from `requirements.txt` using the stdlib `venv` + `pip` workflow — the same workflow used throughout this course.
4. Make an informed choice between venv/pip, conda, and uv for a new project, based on its dependency profile.

---

## Background: the three concerns you are always managing

Before comparing tools, it helps to be clear about the three separate problems they address.

**The interpreter.** Python itself — the executable that runs `.py` files. A machine may have several Python versions installed simultaneously. Some tools (notably `uv`) can also manage interpreter versions; others assume a system or user-installed interpreter and stop there.

**The virtual environment.** An isolated directory tree that holds a private copy of the standard library stubs, a reference back to a specific interpreter, and its own `site-packages` for installed third-party libraries. The point is isolation: changing packages in one project's environment does not affect any other project's environment, and does not touch the system-wide Python install. Without isolation, every project on a machine shares one pool of packages, and version conflicts are guaranteed as soon as any two projects need different versions of the same library.

**The package installer / resolver.** The tool that reads a specification (a `requirements.txt`, a `pyproject.toml`, a conda `environment.yml`) and fetches, builds, and installs packages into a target environment. This is a harder problem than it looks: packages have transitive dependencies, version ranges must be jointly satisfied across all packages at once, and on some platforms the "right" package is a pre-compiled binary wheel whose build tags must match your OS/CPU/Python ABI.

The confusion in the ecosystem is largely because different tools bundle different combinations of these three concerns.

---

## The tools, honestly compared

### `venv` — the stdlib baseline

`venv` is part of the Python standard library since Python 3.3. It creates a virtual environment: an isolated directory with a private `site-packages` and a `python` / `pip` pointing into it.

```
python -m venv .venv
```

That is all it does. It does not install packages. It does not manage Python interpreter versions. It is not faster or slower than the alternatives at package resolution because it does not do package resolution. Its only job is to create the isolated directory.

**Strengths:** Ships with Python — zero additional installation required. The same command works on Windows, macOS, and Linux. Every CI runner and every learner's machine already has it.

**Weaknesses:** Exactly as described — it only creates the environment. You still need `pip` (or a replacement) to install packages into it. It does not help you switch Python versions.

### `virtualenv` — the predecessor

`virtualenv` predates `venv` (it existed before `venv` was added to the stdlib) and is a third-party package (`pip install virtualenv`). It does essentially the same thing as `venv` but adds a few features: faster environment creation, support for Python 2 (long irrelevant), and more configuration surface. In practice, for a modern Python 3 project, the difference is small. Many tools (notably `tox`) use `virtualenv` internally because it has a richer API. You will rarely need to reach for it directly; `venv` is simpler and already present.

### `pip` — the standard installer

`pip` is the standard Python package installer. It reads a requirements specification, contacts PyPI (the Python Package Index), resolves dependencies, downloads packages, and installs them. Starting with Python 3.4, `pip` ships bundled with Python. If you created a `venv`, the `pip` inside it installs into that environment only.

```
# Activate the venv, then:
pip install -r requirements.txt
```

**Strengths:** Universal. Every package on PyPI is pip-installable. Pinned `requirements.txt` files (`package==1.2.3`) are human-readable, trivially diff-able, and understood by every Python toolchain.

**Weaknesses:** Dependency resolution is slower and historically less robust than newer alternatives (pip has improved significantly since 2020 with its backtracking resolver, but it still resolves at install time rather than lock-file time). Binary wheel compatibility has always been pip's weakest area on platforms where pre-built wheels do not exist (e.g. some ARM Linux environments, GPU-specific CUDA builds not covered by the standard wheel matrix).

### `conda` — the cross-language environment manager

`conda` (from Anaconda / Miniconda) is a different animal. It manages environments and installs packages, but it is not Python-specific: `conda` can install R packages, compiled C libraries, CUDA toolkits, and system-level dependencies alongside Python packages. Its package channels (most commonly `conda-forge`) provide pre-built binaries for a wider range of platforms than PyPI wheels.

```
conda create -n myenv python=3.12
conda activate myenv
conda install -c conda-forge numpy opencv
```

**Strengths:** Excellent for projects that need non-Python binaries — CUDA, MKL, BLAS, LAPACK, or platform-specific compiled extensions that are painful to build from source. If your CV project ever needs a custom CUDA toolkit version alongside PyTorch, `conda` installs both the Python package and the native CUDA libraries together.

**Weaknesses:** `conda` environments are heavier and slower to create than `venv` environments. The Anaconda channel historically had licensing restrictions for commercial use (Miniconda/miniforge are free alternatives). Mixing `conda` and `pip` installs into the same environment is a known source of subtle dependency corruption: `conda` does not know about pip-installed packages when it resolves its own graph, so it can overwrite or conflict with them. If you must mix, install everything you can from conda first, then pip-install only what conda cannot provide. Finally, `conda` is an extra install step for a learner who does not already have it — not a blocker, but a friction point.

**When to reach for it:** When you need CUDA, BLAS, or other native library stacks that are difficult to install via pip wheels. For a pure-Python project or one where pip wheels cover your platform, `conda` adds complexity without benefit.

### `uv` — the fast Rust-based tool

`uv`, from Astral (the creators of the `ruff` linter), is a relatively new tool (2024+) that reimplements the pip + venv workflow in Rust. Per Astral's documentation, `uv` resolves and installs packages significantly faster than `pip` for cold installs, because it is written in Rust and uses aggressive caching and parallel downloads. It is pip-compatible: it reads `requirements.txt` and `pyproject.toml`, writes lockfiles (`uv.lock`), and can create and manage virtual environments. Newer versions also manage Python interpreter versions (`uv python install 3.12`), overlapping with tools like `pyenv`.

```
# uv must be installed first (curl-pipe or a system package — not part of Python itself)
uv venv .venv
uv pip install -r requirements.txt
# or with uv's own lockfile workflow:
uv sync
```

**Strengths:** Speed, especially for CI or environments where the package cache is cold. Lockfiles (`uv.lock`) are more reproducible than a bare `requirements.txt` because they pin every transitive dependency, not just direct ones. Active development; the tool is improving quickly.

**Weaknesses:** `uv` is a separate installation step — it is not part of the Python standard library. On a fresh machine (a learner's laptop, a bare CI runner, a compute cluster where you lack admin rights), you must install `uv` before you can use it. It also does not manage non-Python system libraries the way `conda` does, so CUDA-level binary stacks still require another mechanism.

**When to reach for it:** When iteration speed matters — lots of package installs during development, or a CI pipeline where install time is a significant fraction of total run time. Also when you want a first-class lockfile workflow without adopting a full build tool like `Poetry` or `Hatch`.

---

## What this project chose and why

This project uses **`venv` + pinned `requirements.txt`**, for reasons grounded in the teaching mandate.

The decision (recorded in `DEV-LOG.md`, 2026-06-22, and in `PROJECT-PITCH.md`) was: every Python 3.3+ installation ships `venv`, and there is nothing slower about `pip install -r requirements.txt` for a project of this size. A learner should not have to install a tool to install the tools. `uv` and `conda` are the right answers to real problems — speed at scale, and native library stacks — but this pipeline does not have those problems, and introducing either would make the setup instructions harder to follow without providing any benefit the learner can actually observe.

The actual setup commands, taken directly from `requirements.txt`:

```
python -m venv .venv

# Windows:
.venv\Scripts\python.exe -m pip install -r requirements.txt

# macOS / Linux:
.venv/bin/python -m pip install -r requirements.txt
```

Notice: the `pip` that installs packages is the one *inside* the environment (`.venv/Scripts/python.exe -m pip`), not a system-wide `pip`. This ensures every installed package lands in the isolated environment and cannot touch the system install.

The installed packages for the pipeline (from `requirements.txt`) are pinned to exact versions:

```
numpy==2.5.0
pillow==12.2.0
opencv-python-headless==4.13.0.92
fastapi==0.138.0
uvicorn[standard]==0.49.0
pydantic==2.13.4
pydantic-settings==2.14.2
python-multipart==0.0.32
httpx==0.28.1
pytest==9.1.1
```

Two choices here are worth noting. `opencv-python-headless` rather than `opencv-python` — the pipeline is a server and batch processor; it has no GUI and no need for OpenCV's display windows. The headless variant has no GTK/Qt dependency, which makes it lighter and simpler to install in server environments. And "not installed yet" items (`onnxruntime`, `torch`, `imagehash`) are consciously deferred until the research and pipeline stage that justifies each one — the conservative path, not a wish list.

**Citation:** see `research/RESEARCH.md`, entry "Python environment & dependency management (venv/virtualenv/pip/conda/uv) — 2026-06-22" for the official documentation sources.

---

## Failure modes

**Global-install pollution.** If you run `pip install numpy` without activating a virtual environment, the package lands in your system Python's `site-packages`. The next project you work on may pin a different version of numpy. The one after that may install a version that conflicts with something already there. At some point the system Python becomes an unreliable tangle. The fix is not to remember to clean up — it is to never install project packages globally in the first place.

**Unpinned dependencies.** `pip install numpy` installs the *latest* version of numpy at that moment. `requirements.txt` entries without a pinned version (e.g. `numpy>=1.21`) install the latest version satisfying the constraint at that moment. Both mean your environment today and your environment six months from now install different software. For a course, this is particularly damaging: a learner who takes the course a year after it was written should get the same results as a learner today. Pin exact versions.

**Mixing conda and pip.** If you create a conda environment and then pip-install packages into it, conda's resolver no longer has visibility into those packages. The next `conda install` or `conda update` can write over pip-installed files or install conflicting versions because it does not see the pip-managed graph. This is a known, documented conda limitation. The practical rule: if you are in a conda environment and need something pip-installed, do it last, and do not run `conda install` afterward.

**OS-specific wheels.** `pip` installs pre-built binary wheels when they exist for your platform. When they do not (unusual platform, unusual Python version, or a package that requires a native build), `pip` falls back to building from source, which requires a compiler and potentially native headers. On Linux this is usually fine; on Windows it can fail silently or noisily depending on which MSVC version is installed. Symptoms: the install appears to succeed but the import fails because a `.dll` dependency is missing. Mitigation: use a well-supported Python version (3.10–3.12 currently), prefer packages that publish universal wheels, and consult the package's own installation guidance when things go wrong.

---

## Further reading

- **Python `venv` documentation** — https://docs.python.org/3/library/venv.html — the canonical reference; see especially "How venvs work" for the mechanics of the `pyvenv.cfg` file and interpreter resolution.
- **pip User Guide** — https://pip.pypa.io/en/stable/user_guide/ — covers requirements files, pinning, hash verification, and the difference between `pip install -r` and `pip install -e`.
- **conda documentation** — https://docs.conda.io/projects/conda/en/stable/ — environment creation, channel configuration, and the conda-vs-pip mixing cautions are all covered in the official docs.
- **uv documentation** (Astral) — https://docs.astral.sh/uv/ — includes the speed motivation, lockfile workflow (`uv sync`, `uv.lock`), and Python version management (`uv python install`).
