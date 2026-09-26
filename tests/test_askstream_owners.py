"""Each shared askstream type has one definition, in the module that owns it
(plan §2.3-2.5); every other module imports it."""

from __future__ import annotations

import ast
from pathlib import Path

import pplx_agent_tools.askstream as askstream_pkg

OWNERS = {
    "Stage": "frames",
    "Reconnectable": "frames",
    "Change": "blocks",
    "AnswerPaths": "projections",
}


def _top_level_names(tree: ast.Module) -> set[str]:
    out: set[str] = set()
    for node in tree.body:
        match node:
            case ast.ClassDef(name=n) | ast.FunctionDef(name=n):
                out.add(n)
            case ast.Assign(targets=targets):
                out.update(t.id for t in targets if isinstance(t, ast.Name))
            case ast.AnnAssign(target=ast.Name(id=n)):
                out.add(n)
            case _:
                pass
    return out


def test_each_shared_type_is_defined_once_in_its_owner() -> None:
    root = Path(askstream_pkg.__file__).parent
    defined: dict[str, list[str]] = {name: [] for name in OWNERS}
    for path in sorted(root.glob("*.py")):
        names = _top_level_names(ast.parse(path.read_text()))
        for name in OWNERS:
            if name in names:
                defined[name].append(path.stem)
    assert defined == {name: [owner] for name, owner in OWNERS.items()}
