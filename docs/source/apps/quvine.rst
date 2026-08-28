
#######################################
QuVINE
#######################################

**Quantum View-based Network Embeddings**

QuVINE turns a graph into low-dimensional node embeddings by combining
classical and quantum random walks with SGNS-based representation learning. It
is vendored into QBioCode as an in-tree app (``qbiocode.apps.quvine``).

.. _quvine-installation:

Installation
============

QuVINE's Python modules ship with QBioCode, but its third-party dependencies
(gensim, hiperwalk, node2vec, torch-geometric, python-louvain, ripser,
omegaconf) are heavy, so they are behind an optional extra:

.. code-block:: bash

    pip install "qbiocode[quvine]"

A plain ``pip install qbiocode`` leaves QuVINE's dependencies out. ``import
qbiocode`` and every classical embedding still work; requesting a QuVINE method
then raises a :class:`~qbiocode.apps.quvine.QuvineDependencyError` that names
the extra, the missing module, and the command above. To check an existing
environment:

.. code-block:: python

    from qbiocode.apps.quvine import describe_environment
    print(describe_environment())

🔬 **What QuVINE Does**
   1. **Builds multiple views** of a single input graph
   2. **Runs walks**: random walk with restart (RWR) and discrete-/continuous-time quantum walks (DTQW/CTQW)
   3. **Learns embeddings** via skip-gram negative sampling (SGNS), with quantum-calibrated filter / GAT / GraphGPS variants
   4. **Compares against classical baselines**: node2vec, NetMF, APPNP
   5. **Fuses views** into a single embedding when requested

.. note::
    Before you start, make sure that you have installed QBioCode correctly by following the  `Installation <https://ibm.github.io/QBioCode/installation.html>`_ guide.

.. important::
    Graph-complexity metrics are **not** part of the embedding app. They live in
    QBioCode's own :func:`qbiocode.evaluate_graph`
    (``qbiocode.evaluation.graph_evaluation``), which you can run on the same
    graph to characterize it.

Usage
=====

QuVINE can be used as a **command-line tool** or as a **Python library**.

Command-Line Interface
----------------------

After installing QBioCode, the ``quvine`` command embeds the nodes of a graph
given as a 2- or 3-column edge list (``source,target[,weight]``):

.. code-block:: bash

   # List all available methods
   quvine --list-methods

   # Embed a graph with the default fused method
   quvine --edgelist edges.csv --method quvine_fused --output out/

   # Use a classical baseline on a tab-separated, weighted edge list
   quvine --edgelist edges.tsv --sep '\t' --weighted --method node2vec --output out/

The tool writes the embedding matrix and a JSON metadata file to the output
directory.

Python Library Usage
--------------------

.. code-block:: python

   import networkx as nx
   from qbiocode.apps.quvine import embed, list_methods

   # Inspect the available methods (83 in total)
   print(list_methods())

   # Build (or load) a graph
   G = nx.karate_club_graph()

   # Embed its nodes; result.embedding has shape (n_nodes, dim)
   result = embed(G, "quvine_fused", base_seed=0)
   print(result.embedding.shape)

The :func:`embed` function accepts the graph, a ``method`` name, and optional
configuration/override arguments (``config``, ``overrides``, ``base_seed``,
``fuse``, ``n_jobs``, ...). It returns an ``EmbedResult`` whose ``embedding``
attribute is the ``(n_nodes, dim)`` embedding matrix.

Evaluating the input graph:

.. code-block:: python

   from qbiocode import evaluate_graph

   metrics = evaluate_graph(G, name="karate")   # -> pandas.DataFrame

As a QBioCode Embedding
-----------------------

QuVINE is also wired into QBioCode's embedding layer, so any of its method names
is usable wherever ``pca``, ``nmf`` or ``umap`` is -- including QProfiler's
``embeddings:`` config list. The call shape is identical:

.. code-block:: python

   from qbiocode import get_embeddings

   X_train_emb, X_test_emb = get_embeddings(
       "quvine_rwr", X_train, X_test, n_components=8
   )

A symmetric k-nearest-neighbour graph is built over the rows of
``vstack([X_train, X_test])`` with edge weight ``1/(1+d)``, embedded with the
requested method, reduced to ``n_components`` if the method returned a different
width, and split back into the train and test blocks.

.. warning::

   QuVINE methods are **transductive**: no QuVINE method has an out-of-sample
   ``transform``, so the test rows take part in building the graph. Test
   *features* therefore influence the geometry; test *labels* never enter at any
   point. ``get_embeddings`` emits one :class:`UserWarning` per call saying so,
   and :func:`qbiocode.is_transductive` lets downstream code branch on it. The
   classical ``spectral`` mode has exactly the same property.

Discovering what is available:

.. code-block:: python

   import qbiocode

   qbiocode.SKLEARN_METHODS           # the classical modes, always present
   qbiocode.QUVINE_HEADLINE_METHODS   # the QuVINE names worth trying first
   qbiocode.QUVINE_METHODS            # all 83; empty if the extra is absent

``netmf``, ``appnp`` and the ``gat_*`` family need nothing beyond the base
install. The rest raise a message naming the missing dependency and the exact
install command -- see :ref:`the installation note above <quvine-installation>`.

Available Methods
=================

``list_methods()`` returns 83 method names spanning several families. Pass
``kind="sgns"``, ``kind="registry"`` or ``kind="fused"`` to narrow the list:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Family
     - Examples
   * - Fused (all walk kinds combined)
     - ``quvine_fused``, ``quvine``, ``fused``, ``quvine_sgns_fused``
   * - QuVINE (quantum-calibrated)
     - ``quvine_rwr``, ``quvine_ctqw_heat``, ``quvine_dtqw_poly``, ...
   * - Filter variants
     - ``filter_rwr_heat``, ``filter_ctqw_poly``, ``filter_dtqw_heat``, ...
   * - GAT variants
     - ``gat_heat``, ``gat_ctqw_poly``, ``gat_dtqw_heat``, ...
   * - GraphGPS variants
     - ``graphgps_*``
   * - Walks
     - ``rwr``, ``ctqw``, ``dtqw``, ``sgns_*``
   * - Classical baselines
     - ``node2vec``, ``netmf``, ``appnp``, ``graphsage``

.. tip::
    Small quantum-walk step counts often work best: high step counts can
    over-mix the walk, so a modest number of steps tends to give comparable or
    better embeddings.

Tutorials
=========

For worked examples, see the :doc:`Tutorials <../tutorials>` page.

.. TODO(phase-7): once ``tutorials/QuVINE/quvine_sc_cd4_vs_cd8.ipynb`` is in the
   tree, link it directly here. Referencing it before the notebook exists would
   leave a dangling :doc: role that fails a ``sphinx-build -W`` run.

.. seealso::
   - :py:mod:`qbiocode.apps.quvine` - App package
   - :py:func:`qbiocode.evaluate_graph` - Graph-complexity metrics
