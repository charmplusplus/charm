========================================
Editing and building the Charm++ manuals
========================================

The Charm++ manuals in this directory are written in reStructuredText (RST,
http://docutils.sourceforge.net/rst.html) and meant to be built with the
sphinx documentation generator (http://www.sphinx-doc.org/). Pre-built
versions are available on readthedocs.io (https://charm.readthedocs.io).

This file describes how the documentation can be edited and built locally.

Building the manual
===================

Sphinx supports building HTML and PDF versions of the manual. For the HTML
version, only Sphinx is required. Creating the PDF manual also requires pdflatex.

Building the HTML manual:

.. code-block:: bash

  $ pip install sphinx
  $ cd charm/doc
  $ sphinx-build . html/
  $ firefox html/index.html


Building the PDF manual:

.. code-block:: bash

  $ pip install sphinx
  $ cd charm/doc
  $ sphinx-build -b latex . latex/
  $ cd latex
  $ make
  $ evince charm.pdf


RST primer
==========

This section gives a brief overview of reStructuredText (RST) with the most
important items for the Charm++ manual. A more comprehensive manual is
available here: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html.

This file itself is written in RST -- take a look at the source if something is unclear.

Lists
-----

- Itemized:

  .. code-block:: none

    - Item 1
    - Item 2
    ...

- Enumerated:

  .. code-block:: none

    #. Item 1
    #. Item 2
    ...

Sections
--------

Sections get defined by underlining their title:

.. code-block:: none

  Section name
  ============

- First level:  ``======``
- Second level: ``-----``
- Third level:  ``~~~~~~``
- Fourth level: ``^^^^^``
- Fifth level:  ``'''''``

The underline has to have the same length as the title itself.


Code
----

- Inline code (similar to ``\texttt{}``):  \`\`int foo()\`\`: ``int foo();``

  - Inline code is not syntax-highlighted.

- Code blocks (similar to ``\begin{alltt} .. \end{alltt}``):

  - Code blocks have syntax highlighting via the pygments
    (http://pygments.org) library.

  - Do not use the default ``::`` highlighting mode, but specify the
    language explicitly: ``.. code-block:: fortran`` (or ``c++``, ``none``, ...)

    .. code-block:: none

      .. code-block:: fortran

        call foo()
        call bar()

  Versions of pygments newer than 2.3.1 will allow specifying ``charmci`` as the
  language for ci-files (instead of using C++).

Figures
-------

.. code-block:: none

  .. figure:: figures/detailedsim_newer.png
    :name: BigNetSim1
    :width: 3.2in

    Figure caption goes here.


Tables
------

Code:

.. code-block:: none

  .. table:: Table caption goes here.
    :name: tableref

    ============= ==================== ========================================================
    C Field Name  Fortran Field Offset Use
    ============= ==================== ========================================================
    maxResidual   1                    If nonzero, termination criteria: magnitude of residual.
    maxIterations 2                    If nonzero, termination criteria: number of iterations.
    solverIn[8]   3-10                 Solver-specific input parameters.
    ============= ==================== ========================================================

Rendered as:

.. table:: Table caption goes here.
  :name: tableref

  ============= ==================== ========================================================
  C Field Name  Fortran Field Offset Use
  ============= ==================== ========================================================
  maxResidual   1                    If nonzero, termination criteria: magnitude of residual.
  maxIterations 2                    If nonzero, termination criteria: number of iterations.
  solverIn[8]   3-10                 Solver-specific input parameters.
  ============= ==================== ========================================================

References
----------

Adding reference labels
~~~~~~~~~~~~~~~~~~~~~~~

Labels to refer to tables and figures are created by the ``:name:`` property above.
Create labels for sections like this:

.. code-block:: none

  .. _my-label:
  Section ABCD
  ============

Section ABCD can now be referenced with ``my-label`` (note the missing ``_``
and ``:`` in the reference).


Referencing labels
~~~~~~~~~~~~~~~~~~

- With number (best for figures & tables): ``:numref:`reference_name```
- With text: ``:ref:`reference_name``` (text will be taken from referenced item automatically)
- With custom text: ``:ref:`Custom text here <reference_name>```

Links
-----

URLs get parsed and displayed as links automatically. For example: https://charm.cs.illinois.edu/

Citations
---------

.. code-block:: none

  This is a reference [Ref]_ .

  .. [Ref] Paper or article reference, URL, ...

Footnotes
---------

.. code-block:: none

  This text has a footnote [1]_

  .. [1] Text of the footnote.
