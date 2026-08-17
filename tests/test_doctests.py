"""Execute the ``sage:``-prompt doctests embedded in :mod:`uconf` docstrings.

Sage's own ``sage -t`` runner is not shipped by every SageMath installation and
CI has no Sage at all, so nothing was running these examples: a docstring could
raise, or silently document a zero element, without any test noticing. They are
executed here through the stdlib :mod:`doctest` module instead, after rewriting
the ``sage:``/``....:`` prompts into the ``>>>``/``...`` prompts it understands.
"""

from __future__ import annotations

import ast
import doctest
import importlib
import inspect
import io
import pathlib
import pkgutil
import re

import pytest

import uconf

_DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)

_SAGE_PROMPT = re.compile(r"^(\s*)sage:", flags=re.MULTILINE)
_SAGE_CONTINUATION = re.compile(r"^(\s*)\.\.\.\.:", flags=re.MULTILINE)

_OPTION_FLAGS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE | doctest.IGNORE_EXCEPTION_DETAIL


def _to_python_prompts(text: str) -> str:
    """Rewrite Sage's doctest prompts into the ones :mod:`doctest` parses."""
    return _SAGE_CONTINUATION.sub(r"\1...", _SAGE_PROMPT.sub(r"\1>>>", text))


class _SagePromptParser(doctest.DocTestParser):
    """A parser that understands ``sage:`` prompts."""

    def parse(self, string, name="<string>"):
        return super().parse(_to_python_prompts(string), name)


def _iter_module_names():
    """Yield every importable module name inside the :mod:`uconf` package."""
    yield uconf.__name__
    for info in pkgutil.walk_packages(uconf.__path__, prefix=f"{uconf.__name__}."):
        yield info.name


def _own_docstrings(module) -> set[str]:
    """Return the docstrings literally written in ``module``'s source file.

    Decorators such as Sage's ``cached_method`` replace a method by a wrapper
    object carrying Sage's own heavily doctested docstring, which
    :class:`doctest.DocTestFinder` happily attributes to us. Comparing against
    the source keeps the run to examples this package actually wrote.
    """
    source_file = getattr(module, "__file__", None)
    if source_file is None:
        return set()
    tree = ast.parse(pathlib.Path(source_file).read_text(encoding="utf-8"))
    return {
        inspect.cleandoc(docstring)
        for node in ast.walk(tree)
        if isinstance(node, _DOCSTRING_OWNERS)
        and (docstring := ast.get_docstring(node, clean=False)) is not None
    }


def _collect_doctests() -> list[doctest.DocTest]:
    """Return every non-empty doctest written in the package, sorted by name."""
    finder = doctest.DocTestFinder(parser=_SagePromptParser(), exclude_empty=True)
    collected = []
    for module_name in sorted(_iter_module_names()):
        module = importlib.import_module(module_name)
        own = _own_docstrings(module)
        collected.extend(
            test
            for test in finder.find(module, module_name)
            if test.examples and test.docstring and inspect.cleandoc(test.docstring) in own
        )
    return sorted(collected, key=lambda test: test.name)


_DOCTESTS = _collect_doctests()


def test_doctests_are_collected() -> None:
    """Guard against the runner silently finding nothing to execute."""
    assert _DOCTESTS, "No doctests collected; the sage: prompt rewriting is misconfigured."


@pytest.mark.parametrize("test", _DOCTESTS, ids=lambda test: test.name)
def test_docstring_examples(test: doctest.DocTest) -> None:
    runner = doctest.DocTestRunner(optionflags=_OPTION_FLAGS)
    report = io.StringIO()
    runner.run(test, out=report.write)
    assert runner.failures == 0, f"Doctest failures in {test.name}:\n{report.getvalue()}"
