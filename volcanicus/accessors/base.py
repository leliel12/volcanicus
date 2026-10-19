#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2026, Rosenfeld, Hernán; Cabral, Juan B.
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Accessor base class."""

# =============================================================================
# IMPORTS
# =============================================================================

import abc

# =============================================================================
# ACESSOR ABC
# =============================================================================

# Marks a class attribute as abstract: subclasses that don't override it
# can't be created (see AccessorABC.__init_subclass__).
_ABSTRACT = property(abc.abstractmethod(lambda: ...))


class AccessorABC(abc.ABC):
    """Generalization of the accessor idea for use in volcanicus.

    Instances are callable: the first parameter, ``kind``, names the public
    method to run, and every other argument is passed on to it. Without a
    ``kind`` (or when the first argument isn't a string) the method named by
    the class attribute ``_default_kind`` runs instead, so
    ``volcano.episodes(3.0)`` is ``volcano.episodes.find(3.0)``.

    Every subclass must define ``_default_kind``.

    """

    #: Default method to execute.
    _default_kind = _ABSTRACT

    def __init_subclass__(cls):
        """Validate the creation of a subclass."""
        if cls._default_kind is _ABSTRACT:
            raise TypeError(f"{cls!r} must define a _default_kind")

    def __call__(self, kind=None, *args, **kwargs):
        """x.__call__() <==> x()."""
        if kind is None:
            kind = self._default_kind
        elif not isinstance(kind, str):
            # Not a method name: the first argument of the default method.
            args = (kind, *args)
            kind = self._default_kind

        if kind.startswith("_"):
            raise ValueError(f"Invalid kind name '{kind}'")

        method = getattr(self, kind, None)
        if not callable(method):
            raise ValueError(f"Invalid kind name '{kind}'")

        return method(*args, **kwargs)
