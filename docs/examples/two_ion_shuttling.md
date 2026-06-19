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

# Two-Ion Shuttling

This example simulates two distinguishable 1D ion coordinates in one moving harmonic well with softened Coulomb
repulsion.

From the repository root, the console runner writes arrays, summaries, and plots to `outputs/shuttling_two_ion/`:

```bash
python scripts/shuttling/two_ion_shuttling.py \
  --trail-time 4 \
  --grid-points 96 \
  --initial-state localized_crystal_ground_state
```

The `localized_crystal_ground_state` initializer builds a correlated left-right state from the local COM/stretch
normal modes around the ordered classical equilibrium. This is useful when the ion labels should represent stable
transport worldlines rather than the exchange-symmetric collective ground state.

```{code-cell} ipython3
from mqt.yaqs.applications.shuttling import SpatialGrid, TwoIonShuttlingConfig, simulate_two_ion_shuttling

result = simulate_two_ion_shuttling(
    TwoIonShuttlingConfig(
        grid=SpatialGrid(x_min=-8.0, x_max=8.0, num_points=72),
        q_initial=-1.5,
        q_final=1.5,
        duration=3.0,
        trail_time=3.0,
        dt=0.05,
        snapshot_stride=5,
        initial_state="localized_crystal_ground_state",
        mode_population_levels=4,
        com_mode_population_levels=16,
    )
)

print(f"final excess energy / (hbar omega): {result.excess_quanta[-1]:.3e}")
print(f"final COM-like excitation: {result.com_mode_excitation[-1]:.3e}")
print(f"final stretch-like excitation: {result.stretch_mode_excitation[-1]:.3e}")
print(f"final mean separation: {result.mean_separation[-1]:.3e}")
```

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
from mqt.yaqs.applications.shuttling.visualization import plot_center_of_mass_trace

fig, ax = plot_center_of_mass_trace(result)
fig
```

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
from mqt.yaqs.applications.shuttling.visualization import plot_excess_energy_trace

fig, ax = plot_excess_energy_trace(result)
fig
```

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
from mqt.yaqs.applications.shuttling.visualization import plot_mode_excitation_trace

fig, ax = plot_mode_excitation_trace(result)
fig
```

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
from mqt.yaqs.applications.shuttling.visualization import plot_mode_populations

fig, axes = plot_mode_populations(result)
fig
```

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
from mqt.yaqs.applications.shuttling.visualization import plot_joint_mode_populations

fig, axes = plot_joint_mode_populations(result)
fig
```

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
from mqt.yaqs.applications.shuttling.visualization import plot_mode_population_consistency

fig, axes = plot_mode_population_consistency(result)
fig
```

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
from mqt.yaqs.applications.shuttling.visualization import plot_two_ion_phase_space

fig, ax = plot_two_ion_phase_space(result)
fig
```

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
from mqt.yaqs.applications.shuttling.visualization import plot_normal_mode_phase_spaces

fig, axes = plot_normal_mode_phase_spaces(result)
fig
```

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
from mqt.yaqs.applications.shuttling.visualization import plot_pair_density_snapshot

fig, ax = plot_pair_density_snapshot(result)
fig
```
