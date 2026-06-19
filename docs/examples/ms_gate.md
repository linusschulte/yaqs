---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
  execution_timeout: 300
---

```{code-cell} ipython3
:tags: [remove-cell]
%config InlineBackend.figure_formats = ['svg']
```

# Analog MS Gate

This example runs a standalone two-qubit Mølmer-Sørensen gate in a truncated COM/stretch collective motional basis.
It is independent of the shuttling primitive: the motional input is initialized directly in the collective basis.

From the repository root, the console runner writes arrays, summaries, and plots to `outputs/ms_gate/`:

```bash
python scripts/shuttling/ms_gate.py
```

The finite-Lamb-Dicke sideband model is calibrated to the cold motional ground state and then reused for hotter COM,
stretch, and mixed inputs.

```{code-cell} ipython3
from dataclasses import replace

import numpy as np

from mqt.yaqs.applications.shuttling import (
    collective_diagonal_mixture,
    collective_fock_state,
    collective_ground_state,
    simulate_ms_gate,
)

cold = simulate_ms_gate()
config = replace(cold.config, drive_amplitude=cold.calibrated_drive_amplitude, calibrate=False)

populations = np.zeros((12, 12), dtype=float)
populations[0, 0] = 0.5
populations[5, 0] = 0.25
populations[0, 8] = 0.25

results = {
    "cold": cold,
    "COM excited": simulate_ms_gate(config, collective_fock_state(12, 12, 5, 0)),
    "stretch excited": simulate_ms_gate(config, collective_fock_state(12, 12, 0, 8)),
    "mixed": simulate_ms_gate(config, collective_diagonal_mixture(populations)),
}

for name, result in results.items():
    print(f"{name:16s} final fidelity = {result.target_fidelity[-1]:.6f}")
```

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
from mqt.yaqs.applications.shuttling.visualization import plot_ms_final_fidelities

fig, ax = plot_ms_final_fidelities(results)
fig
```

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
from mqt.yaqs.applications.shuttling.visualization import plot_ms_fidelity

fig, ax = plot_ms_fidelity(results["cold"])
fig
```

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
from mqt.yaqs.applications.shuttling.visualization import plot_ms_spin_populations

fig, ax = plot_ms_spin_populations(results["cold"])
fig
```

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
from mqt.yaqs.applications.shuttling.visualization import plot_ms_mode_expectations

fig, ax = plot_ms_mode_expectations(results["COM excited"])
fig
```

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
from dataclasses import replace

from mqt.yaqs.applications.shuttling.visualization import plot_ms_phase_space_loops

imperfect_config = replace(config, duration=0.85 * config.duration)
imperfect = simulate_ms_gate(imperfect_config, collective_ground_state(12, 12))

fig, axes = plot_ms_phase_space_loops({"closed": results["cold"], "short gate": imperfect})
fig
```
