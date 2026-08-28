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

"""Graph and dataset preparation helpers for QuVINE.

.. warning::

   This package is **incomplete**. Four modules that the rest of QuVINE imports
   were never committed to the internal repository: an *unanchored* ``data/``
   rule in its ``.gitignore`` matched ``qbiocode/apps/quvine/data/`` at every
   depth, so the directory was silently excluded and never entered git history.
   (External's ``.gitignore`` anchors the rule as ``/data/`` and does not have
   this problem, which is why the directory can be tracked here.)

   Present and working:

   * :mod:`~qbiocode.apps.quvine.data.subgraph` -- reimplemented here from its
     call sites, since ego-net expansion is a well-determined graph primitive
     and the whole 69-method registry depends on it.

   Still missing, with the feature each one gates:

   =============================== =========================================
   Module                          Feature it gates
   =============================== =========================================
   ``data_loader``                 :class:`~qbiocode.apps.quvine.Pipeline`
                                   loading graphs and GWAS tables from disk
   ``prepare``                     ``Pipeline`` graph preprocessing
   ``random_graphs``               synthetic benchmark generators used by
                                   ``reproducibility.graph_generator``
   ``random_graphs_extended``      SBM variants used by the same
   =============================== =========================================

   Everything reachable from :func:`qbiocode.get_embeddings` and the ``quvine``
   CLI works without them. Dropping the original files into this directory needs
   no other change -- the import paths that reference them are unmodified.
"""

import importlib

from qbiocode.apps.quvine.data.subgraph import expand_neighborhood

__all__ = ["QuvineDataUnavailableError", "expand_neighborhood", "require_data_module"]


class QuvineDataUnavailableError(ImportError):
    """A module of this package is referenced but absent from the checkout.

    Subclasses :class:`ImportError` so the existing ``except ImportError`` guards
    around optional QuVINE features keep working.
    """


def require_data_module(module_name: str, feature: str):
    """Import ``qbiocode.apps.quvine.data.<module_name>`` or explain its absence.

    Args:
        module_name: Bare module name inside this package, e.g. ``"data_loader"``.
        feature: What the caller was trying to do, named the way a user would
            recognise it, e.g. ``"Pipeline graph loading"``.

    Returns:
        The imported module.

    Raises:
        QuvineDataUnavailableError: if the module is one of the four that were
            never committed. The message says which module, which feature is
            therefore unavailable, and what to do about it.
        ModuleNotFoundError: unchanged, if the module exists but one of *its*
            third-party imports is missing -- that is a dependency problem and
            must not be reported as missing source.
    """
    qualified = f"{__name__}.{module_name}"
    try:
        return importlib.import_module(qualified)
    except ModuleNotFoundError as exc:
        if exc.name != qualified:
            # Something deeper is missing; do not blame our own source tree.
            raise
        raise QuvineDataUnavailableError(
            f"{feature} needs {qualified}, which is not present in this "
            f"installation. That module was never committed to the QuVINE source "
            f"repository -- an unanchored 'data/' rule in its .gitignore excluded "
            f"the whole directory. Everything reachable from "
            f"qbiocode.get_embeddings() and the 'quvine' CLI works without it; see "
            f"the qbiocode.apps.quvine.data docstring for the full list of what "
            f"this gates."
        ) from exc
