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

# Single-Ion Shuttling

This example simulates a single ion in a one-dimensional moving harmonic potential.
The dynamics are closed-system and motional only: no spin, Coulomb coupling, bath, or gate dynamics are included.
To generate the standard no-shuttle, slow-shuttle, and fast-shuttle outputs from the console, run:

```bash
python scripts/shuttling/single_ion_shuttling.py
```

By default, the runner writes arrays, summaries, static plots, and a fast-shuttle animation to `outputs/shuttling/`.
Add `--trail-time 2.0` to keep simulating after each trajectory has reached its final trap center.

```{code-cell} ipython3
from mqt.yaqs.applications.shuttling import ShuttlingConfig, SpatialGrid, simulate_shuttling

grid = SpatialGrid(x_min=-8.0, x_max=8.0, num_points=256)

no_shuttle = simulate_shuttling(
    ShuttlingConfig(grid=grid, q_initial=0.0, q_final=0.0, duration=2.0, dt=0.02, population_levels=4)
)
slow_shuttle = simulate_shuttling(
    ShuttlingConfig(grid=grid, q_initial=-1.5, q_final=1.5, duration=8.0, dt=0.02, population_levels=4)
)
fast_shuttle = simulate_shuttling(
    ShuttlingConfig(
        grid=grid,
        q_initial=-1.5,
        q_final=1.5,
        duration=1.0,
        trail_time=2.0,
        dt=0.01,
        population_levels=4,
    )
)

print(f"no shuttle final <n>: {no_shuttle.excitation_number[-1]:.3e}")
print(f"slow shuttle final <n>: {slow_shuttle.excitation_number[-1]:.3e}")
print(f"fast shuttle final <n>: {fast_shuttle.excitation_number[-1]:.3e}")
```

Plot the trap center and wave-packet center for the fast shuttle.

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
from mqt.yaqs.applications.shuttling.visualization import plot_position_trace

fig, ax = plot_position_trace(fast_shuttle)
fig
```

Plot the instantaneous motional excitation number, computed relative to the harmonic well centered at the current trap
position.

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
from mqt.yaqs.applications.shuttling.visualization import plot_excitation_trace

fig, ax = plot_excitation_trace(fast_shuttle)
fig
```

Plot the momentum expectation value and spread.

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
from mqt.yaqs.applications.shuttling.visualization import plot_momentum_trace

fig, ax = plot_momentum_trace(fast_shuttle)
fig
```

Plot the phase-space trajectory relative to the moving trap center.

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
from mqt.yaqs.applications.shuttling.visualization import plot_phase_space

fig, ax = plot_phase_space(fast_shuttle)
fig
```

The first few instantaneous harmonic-oscillator populations are available when `population_levels` is nonzero.

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
from mqt.yaqs.applications.shuttling.visualization import plot_populations

fig, ax = plot_populations(fast_shuttle)
fig
```
