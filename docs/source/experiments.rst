experiments package
===================

Scripts that replicate the numerical experiments from Mohajerin Esfahani & Kuhn (2018).

Run all experiments
-------------------

.. code-block:: bash

   python experiments/run_all.py          # full run (200 simulations)
   python experiments/run_all.py --quick  # quick test (5 simulations)
   python experiments/run_all.py --fig 4  # single figure

Figure 4: Portfolio Composition
-------------------------------

.. automodule:: experiments.fig4_portfolio_composition
   :members:
   :show-inheritance:
   :undoc-members:

Figure 5: Out-of-Sample Performance
------------------------------------

.. automodule:: experiments.fig5_out_of_sample
   :members:
   :show-inheritance:
   :undoc-members:

Figure 6: Performance Comparison
---------------------------------

.. automodule:: experiments.fig6_performance_comparison
   :members:
   :show-inheritance:
   :undoc-members:

Figure 8: Wasserstein Radius vs N
----------------------------------

.. automodule:: experiments.fig8_radius_vs_N
   :members:
   :show-inheritance:
   :undoc-members: