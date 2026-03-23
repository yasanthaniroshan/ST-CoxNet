"""
Wrapper for v26 FiLM one-stage experiment.

Runs:
  `cpc_train_v26_film_conditional_patient_context_mean_relative_rr.py`
"""

import runpy

runpy.run_path(
    "cpc_train_v26_film_conditional_patient_context_mean_relative_rr.py",
    run_name="__main__",
)

