.. _configure:

Configuring OPOF
================

Setting Concurrency
-------------------

By default, OPOF uses all available CPUs for training. To limit the number of CPUs per training instance (e.g. to 16), execute:

.. code-block:: console

   $ echo 'export OPOF_CONCURRENCY=16' >> ~/.bashrc

Remember to re-source the environment for changes to take effect:

.. code-block:: console

   $ source ~/.bashrc

External Domains and Algorithms
-------------------------------

External packages which contain custom domains and algorithms may be used alongside the core OPOF package.
To install an external package, please refer to the instructions provided by the specific package.

After installing an external package, OPOF needs to be configured to discover the newly added domains and 
algorithms. 

**1. Register domains**

If the external package contains newly exposed domains, execute

.. code-block:: console

   $ echo 'export OPOF_DOMAINS="<external_package.domains>:$OPOF_DOMAINS"' >> ~/.bashrc

replacing ``<external_package.domains>`` with the module path where the domains are exposed.

**2. Register algorithms**

If the package contains newly exposed algorithms, execute

.. code-block:: console

   $ echo 'export OPOF_ALGORITHMS="<external_package.algorithms>:$OPOF_ALGORITHMS"' >> ~/.bashrc

replacing ``<external_package.algorithms>`` with the module path where the algorithms are exposed.

**3. Source environment**

Once you have registered the external domains and/or algorithms, execute

.. code-block:: console

   $ source ~/.bashrc

to re-source the environment.

**4. Verify environment**

To check that OPOF can discover your external domains and algorithms corrently, execute

.. code-block:: console

   $ opof-registry

which should print the list of domains and algorithms discovered by OPOF.
