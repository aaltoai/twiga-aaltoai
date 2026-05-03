"""Shared fixtures for twiga-aaltoai tests."""

import os
import sys
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Mock the *entire* `app` package that twiga exposes so that tests which
# import eval.answer_generator (which in turn imports from app.*) never
# trigger the real twiga settings initialisation (requires DB URL, Meta
# secrets, etc.).
#
# We insert stub modules into sys.modules BEFORE any test file imports
# answer_generator.  conftest.py is loaded first by pytest, so this works.
# ---------------------------------------------------------------------------

def _install_twiga_stubs() -> dict[str, MagicMock]:
    """Insert lightweight stubs for twiga's `app.*` modules into sys.modules."""
    stubs: dict[str, MagicMock] = {}

    # -- app (top-level) ---------------------------------------------------
    app_mod = MagicMock()
    stubs["app"] = app_mod

    # -- app.config --------------------------------------------------------
    config_mod = MagicMock()
    # Provide concrete objects that pages/answer_generator read
    config_mod.settings = MagicMock(debug=False)
    config_mod.llm_settings = MagicMock(
        llm_name="test-model",
        provider=MagicMock(value="test-provider"),
    )
    config_mod.embedding_settings = MagicMock(
        embedder_name="test-embedder",
        provider=MagicMock(value="test-embed-provider"),
    )
    stubs["app.config"] = config_mod

    # -- app.database ------------------------------------------------------
    db_mod = MagicMock()
    stubs["app.database"] = db_mod

    # -- app.database.models -----------------------------------------------
    models_mod = MagicMock()
    # Create lightweight classes so that answer_generator can instantiate them
    for cls_name in ("User", "Message", "TeacherClass", "Class", "Subject"):
        klass = type(cls_name, (), {"__init__": lambda self, **kw: self.__dict__.update(kw)})
        setattr(models_mod, cls_name, klass)
    stubs["app.database.models"] = models_mod

    # -- app.database.enums ------------------------------------------------
    from enum import Enum

    class UserState(str, Enum):
        active = "active"
        inactive = "inactive"

    class MessageRole(str, Enum):
        user = "user"
        assistant = "assistant"
        system = "system"

    class GradeLevel(str, Enum):
        os2 = "os2"

    class SubjectName(str, Enum):
        geography = "geography"

    enums_mod = MagicMock()
    enums_mod.UserState = UserState
    enums_mod.MessageRole = MessageRole
    enums_mod.GradeLevel = GradeLevel
    enums_mod.SubjectName = SubjectName
    stubs["app.database.enums"] = enums_mod

    # -- app.services ------------------------------------------------------
    services_mod = MagicMock()
    stubs["app.services"] = services_mod

    # -- app.services.llm_service ------------------------------------------
    llm_service_mod = MagicMock()
    llm_service_mod.llm_client = MagicMock()
    stubs["app.services.llm_service"] = llm_service_mod

    # Install into sys.modules
    for name, mod in stubs.items():
        sys.modules[name] = mod

    return stubs


# Install stubs at conftest import time (before any test collection)
_twiga_stubs = _install_twiga_stubs()


@pytest.fixture
def twiga_stubs():
    """Provide access to the mock stubs for twiga's app.* modules."""
    return _twiga_stubs
