# Copyright 2026, IBM Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The Sphinx site builds, and builds *structurally* clean.

Deliberately not ``-W``. Turning every warning into an error makes the docs job
fail on an unrelated docstring nit in a module nobody touched, which is how
``continue-on-error: true`` ended up on the job in the first place. Instead the
build must succeed, and must emit none of the warnings that mean the site is
actually broken: a toctree pointing at a document that does not exist, a page
reachable from nothing, an unresolvable reference.

Skipped without the ``[docs]`` extra, without the pandoc binary that nbsphinx
shells out to for notebook pages, and without nbconvert's ``rst`` template --
those three are missing toolchain, not broken documentation, and reporting them
as failures would train people to ignore this test.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from .conftest import REPO_ROOT, subprocess_env

pytest.importorskip("sphinx", reason="the [docs] extra is not installed")

DOCS_SOURCE = REPO_ROOT / "docs" / "source"

# Warnings that mean the published site is wrong, as opposed to untidy. Each is
# matched against the full warning text, case-insensitively.
STRUCTURAL_WARNINGS = (
    r"toctree contains reference to nonexisting document",
    r"toctree contains reference to excluded document",
    r"document isn't included in any toctree",
    r"undefined label",
    r"unknown document",
    r"unknown source document",
    r"image file not readable",
    r"autodoc: failed to import",
    # An object described twice is indexed twice, which makes every short
    # cross-reference to it ambiguous and sends readers to an arbitrary one.
    r"duplicate object description",
    r"more than one target found for cross-reference",
    # docutils rejects the node outright, so the page renders without it.
    r"Transition must be child of",
)

# A warning whose location is a hand-written page under ``docs/source`` is
# structural too: that page's own markup is wrong, and the reader sees the
# damage. Docstring warnings are deliberately not covered -- they carry a
# ``qbiocode/`` location, there are about thirty of them, and reformatting every
# docstring in the tree is a separate job from this one.
PAGE_WARNING_EXEMPTIONS = (
    # ``.. automodule::`` registers its ``module-<name>`` anchors through the
    # Python domain. MyST's local-id check does not see the Python domain, so it
    # reports every cross-page link to one as missing. The anchors *are* in the
    # rendered HTML and the links resolve -- confirmed by grepping
    # ``api/qbiocode.learning.html`` for ``id="module-qbiocode.learning.*"``.
    r"local id not found in doc",
)

PAGE_WARNING = re.compile(
    r"^(?P<path>\S*[/\\]docs[/\\]source[/\\]\S+?):(?P<line>\d+):\s*"
    r"(?P<level>WARNING|ERROR)\b"
)


def _nbconvert_rst_template_available() -> bool:
    """nbsphinx renders notebook cells through nbconvert's ``rst`` template.

    The template ships as data under a prefix's ``share/jupyter``, and jupyter
    only searches ``sys.prefix``. In a virtualenv built with
    ``--system-site-packages``, nbconvert is importable from the base
    environment while its templates are not on the search path, and the build
    dies in extension setup with "No template sub-directory with name 'rst'".
    That is an environment defect, not a documentation defect, so skip rather
    than report the docs as broken.
    """
    try:
        from jupyter_core.paths import jupyter_path
    except ImportError:
        return False
    return any(
        (Path(directory) / "nbconvert" / "templates" / "rst").is_dir()
        for directory in jupyter_path()
    )


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    """One build, shared by every assertion below -- Sphinx is not cheap."""
    if shutil.which("pandoc") is None:
        pytest.skip("nbsphinx needs the pandoc binary to render notebook pages")
    if not _nbconvert_rst_template_available():
        pytest.skip(
            "nbconvert's rst template is not on the jupyter search path; set "
            "JUPYTER_PATH to the prefix that owns share/jupyter"
        )
    out_dir = tmp_path_factory.mktemp("html")
    doctrees = tmp_path_factory.mktemp("doctrees")
    completed = subprocess.run(
        [
            sys.executable, "-m", "sphinx",
            "-b", "html",
            "-d", str(doctrees),
            str(DOCS_SOURCE), str(out_dir),
        ],
        cwd=str(REPO_ROOT),
        env=subprocess_env(),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    return completed, out_dir


def test_the_build_succeeds(build):
    completed, _ = build
    assert completed.returncode == 0, (
        f"sphinx-build exited {completed.returncode}\n"
        f"--- stdout ---\n{completed.stdout[-4000:]}\n"
        f"--- stderr ---\n{completed.stderr[-4000:]}"
    )


def test_it_emits_no_structural_warnings(build):
    completed, _ = build
    text = completed.stdout + completed.stderr
    offenders = [
        line
        for line in text.splitlines()
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in STRUCTURAL_WARNINGS)
    ]
    assert not offenders, "the site built, but its structure is broken:\n" + "\n".join(
        offenders[:40]
    )


def test_no_hand_written_page_has_a_markup_warning(build):
    """Markup nits in ``docs/source`` pages are visible defects, not noise.

    The structural list above catches warnings by *kind*. This catches them by
    *location*: anything docutils or MyST reports against a file the team wrote
    by hand renders wrong on the site. It is what would have caught two nested
    bullet lists indented past their parent (rendered as block quotes), a
    paragraph indented one space too far, and a Colab cell fenced as ``python``
    whose IPython magics the Python lexer could not tokenise -- none of which
    matched any pattern in ``STRUCTURAL_WARNINGS``.
    """
    completed, _ = build
    text = completed.stdout + completed.stderr
    offenders = [
        line
        for line in text.splitlines()
        if PAGE_WARNING.match(line.strip())
        and not any(
            re.search(pattern, line, re.IGNORECASE)
            for pattern in PAGE_WARNING_EXEMPTIONS
        )
    ]
    assert not offenders, (
        "hand-written pages under docs/source emit markup warnings, so they do "
        "not render as intended:\n" + "\n".join(offenders[:40])
    )


@pytest.mark.parametrize(
    "page",
    [
        "index.html",
        "installation.html",
        "tutorials.html",
        "apps/quvine.html",
        "api/qbiocode.evaluation.html",
    ],
)
def test_the_expected_pages_are_written(build, page):
    _, out_dir = build
    assert (out_dir / page).is_file(), f"{page} was not produced"


def test_the_sphinx_output_lands_where_ci_uploads_from():
    """``docs/Makefile`` and the CI upload path must name the same directory.

    They disagreed once: the Makefile built into ``docs/build`` while a
    hand-made ``docs/_build/html`` tree was committed and published.
    """
    makefile = (REPO_ROOT / "docs" / "Makefile").read_text()
    match = re.search(r"^BUILDDIR\s*=\s*(\S+)", makefile, re.MULTILINE)
    assert match, "docs/Makefile does not define BUILDDIR"
    build_dir = match.group(1)
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert f"docs/{build_dir}/html" in workflow, (
        f"docs/Makefile builds into docs/{build_dir}, which the CI workflow "
        f"never uploads from"
    )
