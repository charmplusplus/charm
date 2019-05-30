Quickstart
==========


Installing Charm++
------------------

To download the latest Charm++ release, run:

.. code-block:: console

   $ git clone https://charm.cs.illinois.edu/gerrit/charm
   $ tar xzf charm-6.9.0.tar.gz

To download the development version of Charm++, run:

.. code-block:: console

   $ git clone https://charm.cs.illinois.edu/gerrit/charm


To build Charm++, use the following commands:

.. code-block:: console

   $ cd charm
   $ ./build AMPI netlrts-linux-x86_64 --with-production -j4

This is the recommened version to install on Linux systems. For MacOS,
substitute "linux" with "darwin". For advanced options, please see
Section :numref:`sec:install` of the manual.


Parallel "Hello World" with Charm++
-----------------------------------

