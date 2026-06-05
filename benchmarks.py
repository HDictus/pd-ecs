#!/usr/bin/env python3
"""
Benchmark suite for pd-ecs core operations.

For each operation × n_entities × n_components combination:
  - Profiles the isolated operation with cProfile → profiles/<op>_<Ne>e_<Nc>c.prof
  - Times the operation over several repeats for the scaling plot

Outputs:
  profiles/scaling.png  — log-log scaling curves per operation
"""

import cProfile
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pd_ecs import Component, World
from pd_ecs._archetype_store import ArchetypeStore

PROFILES_DIR = "profiles"
N_ENTITIES_VALS = [100, 1_000, 10_000, 100_000]
N_COMPONENTS_VALS = [2, 4, 8, 16]
N_TIMING_REPEATS = 5

os.makedirs(PROFILES_DIR, exist_ok=True)


def make_components(n):
    return [Component(f"c{i}") for i in range(n)]


# ---------------------------------------------------------------------------
# Benchmark factories
# Each returns a (setup, run) pair:
#   setup() -> fresh state
#   run(state) -> performs the single operation under test (no return value needed)
# ---------------------------------------------------------------------------

def bm_add_entities(n_entities, components):
    data = {c: np.random.randn(n_entities) for c in components}

    def setup():
        return World()

    def run(world):
        world.add_entities(data)

    return setup, run


def bm_remove_entities(n_entities, components):
    ids = list(range(n_entities))

    def setup():
        world = World()
        world.add_entities({c: np.random.randn(n_entities) for c in components})
        return world

    def run(world):
        world.remove_entities(ids)

    return setup, run


def bm_give(n_entities, components):
    """Give a new component to half the existing entities (archetype migration)."""
    base_comps = components[:-1]
    new_comp = components[-1]
    ids = list(range(n_entities // 2))
    values = {new_comp: np.random.randn(len(ids))}

    def setup():
        world = World()
        world.add_entities({c: np.random.randn(n_entities) for c in base_comps})
        return world

    def run(world):
        world.give(ids, values)

    return setup, run


def bm_take(n_entities, components):
    """Strip one component from half the existing entities (archetype migration)."""
    target_comp = components[-1]
    ids = list(range(n_entities // 2))

    def setup():
        world = World()
        world.add_entities({c: np.random.randn(n_entities) for c in components})
        return world

    def run(world):
        world.take(ids, target_comp)

    return setup, run


def bm_query_single(n_entities, components):
    """Read a single component series."""
    comp = components[0]

    def setup():
        world = World()
        world.add_entities({c: np.random.randn(n_entities) for c in components})
        return world

    def run(world):
        _ = world[comp]

    return setup, run


def bm_query_multi(n_entities, components):
    """Inner-join query across all components."""

    def setup():
        world = World()
        world.add_entities({c: np.random.randn(n_entities) for c in components})
        return world

    def run(world):
        _ = world[components]

    return setup, run


def bm_query_filter(n_entities, components):
    """Query with a negation filter: comp[0] AND NOT comp[-1].

    Half the entities match (they have comp[0] but not comp[-1]).
    """
    half = n_entities // 2
    filt = [components[0], ~components[-1]]

    def setup():
        world = World()
        # First half: all components (won't match the NOT filter)
        world.add_entities({c: np.random.randn(half) for c in components})
        # Second half: all but last component (will match)
        world.add_entities({c: np.random.randn(half) for c in components[:-1]})
        return world

    def run(world):
        _ = world[filt]

    return setup, run


def bm_update(n_entities, components):
    """Overwrite all values for one component via world.update()."""
    comp = components[0]
    new_values = pd.DataFrame(
        {comp: np.random.randn(n_entities)}, index=range(n_entities)
    )

    def setup():
        world = World()
        world.add_entities({c: np.random.randn(n_entities) for c in components})
        return world

    def run(world):
        world.update(new_values)

    return setup, run


def bm_archstore_add_entities(n_entities, components):
    """Low-level ArchetypeStore.add_entities — bulk entity registration."""
    eids = list(range(n_entities))

    def setup():
        return ArchetypeStore()

    def run(archs):
        archs.add_entities(eids)

    return setup, run


def bm_choose_archetypes(n_entities, components):
    """ArchetypeStore.choose_archetypes filter over a populated store."""
    eids = list(range(n_entities))
    filt = [components[0], ~components[-1]]

    def setup():
        archs = ArchetypeStore()
        archs.add_entities(eids)
        # Give half the entities the last component so both archetypes exist
        for comp in components[:-1]:
            archs.add_component(eids, comp)
        archs.add_component(eids[: n_entities // 2], components[-1])
        return archs

    def run(archs):
        _ = archs.choose_archetypes(filt)

    return setup, run


BENCHMARKS = {
    "add_entities": bm_add_entities,
    "remove_entities": bm_remove_entities,
    "give": bm_give,
    "take": bm_take,
    "query_single": bm_query_single,
    "query_multi": bm_query_multi,
    "query_filter": bm_query_filter,
    "update": bm_update,
    "archstore_add_entities": bm_archstore_add_entities,
    "choose_archetypes": bm_choose_archetypes,
}


# ---------------------------------------------------------------------------
# Profiling + timing
# ---------------------------------------------------------------------------

def profile_op(name, setup, run, n_entities, n_components):
    state = setup()
    prof_path = os.path.join(
        PROFILES_DIR, f"{name}_{n_entities}e_{n_components}c.prof"
    )
    pr = cProfile.Profile()
    pr.enable()
    run(state)
    pr.disable()
    pr.dump_stats(prof_path)
    return prof_path


def time_op(setup, run, n_repeats):
    """Return median wall-clock time (seconds) over n_repeats independent runs."""
    times = []
    for _ in range(n_repeats):
        state = setup()
        t0 = time.perf_counter()
        run(state)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def run_benchmarks():
    results = {}  # results[name][(n_entities, n_components)] = seconds

    for name, factory in BENCHMARKS.items():
        print(f"\n=== {name} ===")
        results[name] = {}
        for n_components in N_COMPONENTS_VALS:
            components = make_components(n_components)
            for n_entities in N_ENTITIES_VALS:
                pair = factory(n_entities, components)
                setup, run = pair

                prof_path = profile_op(name, setup, run, n_entities, n_components)
                elapsed = time_op(setup, run, N_TIMING_REPEATS)
                results[name][(n_entities, n_components)] = elapsed

                print(
                    f"  {n_entities:>7} entities  {n_components:>2} components"
                    f"  → {elapsed * 1000:8.3f} ms   [{prof_path}]"
                )

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results):
    n_ops = len(results)
    ncols = 3
    nrows = (n_ops + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, len(N_COMPONENTS_VALS)))

    for ax, (name, data) in zip(axes, results.items()):
        for color, n_components in zip(colors, N_COMPONENTS_VALS):
            xs, ys = [], []
            for n_entities in N_ENTITIES_VALS:
                key = (n_entities, n_components)
                if key in data:
                    xs.append(n_entities)
                    ys.append(data[key] * 1_000)  # → milliseconds
            if xs:
                ax.plot(
                    xs, ys,
                    marker="o", color=color,
                    label=f"{n_components} comps",
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("entities")
        ax.set_ylabel("time (ms)")
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)

    for ax in axes[n_ops:]:
        ax.set_visible(False)

    fig.suptitle("pd-ecs operation scaling", fontsize=13, y=1.01)
    # plt.tight_layout()
    out = os.path.join(PROFILES_DIR, "scaling.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")
    plt.show()


if __name__ == "__main__":
    results = run_benchmarks()
    plot_results(results)
