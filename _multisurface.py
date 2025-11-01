# _multisurface.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Dict

import jax.numpy as jnp
from jax import random

from _state import runtime
from _geometry import (
    surface_points_and_normals,
    inside_torus_mask,
    fixed_box_points,
    select_interior_from_fixed,
)

# If/when you support arbitrary meshes:
# from _geometry_arbitrary import load_surface_xyz, estimate_normals_pca, winding_number_inside

@dataclass
class SurfaceItem:
    name: str
    P_bdry: jnp.ndarray        # [Nb,3]
    N_bdry: jnp.ndarray        # [Nb,3]
    inside_mask_fn: Callable[[jnp.ndarray], jnp.ndarray]  # takes [M,3] -> [M] bool
    shape_thetaphi: Optional[Tuple[int, int]] = None

class SurfacesDataset:
    def __init__(self, items: List[SurfaceItem]):
        self.items = items

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

    def sample_batch(self, key: jnp.ndarray, N_in: int, P_box: jnp.ndarray):
        """Pick a random surface; return (surface, P_in, P_bdry, N_bdry)."""
        idx = int(random.randint(key, shape=(), minval=0, maxval=len(self.items)))
        surf = self.items[idx]
        mask = surf.inside_mask_fn(P_box)  # [M] bool
        valid_ids = jnp.nonzero(mask, size=P_box.shape[0], fill_value=0)[0]
        # <- NEW: randomly choose interior points for this surface/step
        key, sub = random.split(key)
        n_valid = int(mask.sum())                 # Python int
        # sample without replacement up to N_in; fall back to first N if too few
        take = min(N_in, n_valid)                 # Python int
        sel = random.choice(sub, valid_ids[:n_valid], (take,), replace=False)
        P_in = P_box[sel]
        # If fewer than N_in exist (unlikely), pad by repeating
        if P_in.shape[0] < N_in:
            reps = N_in - P_in.shape[0]
            P_in = jnp.concatenate([P_in, P_in[:reps]], axis=0)
        return surf, P_in, surf.P_bdry, surf.N_bdry

# ---------- Builders ----------

def build_torus_surface_item(name: str, R0: float, a0: float, a1: float, N_harm: int,
                             n_theta: int, n_phi: int) -> SurfaceItem:
    from _state import runtime as rt

    # snapshot runtime, then restore
    R0_old, a0_old, a1_old, Nh_old = rt.R0, rt.a0, rt.a1, rt.N_harm
    rt.R0, rt.a0, rt.a1, rt.N_harm = R0, a0, a1, N_harm
    try:
        P_bdry, N_bdry, Xg, Yg, Zg = surface_points_and_normals(n_theta, n_phi)
        def inside_mask(points: jnp.ndarray) -> jnp.ndarray:
            x, y, z = points[:,0], points[:,1], points[:,2]
            return inside_torus_mask(x, y, z)
        return SurfaceItem(name=name, P_bdry=P_bdry, N_bdry=N_bdry, inside_mask_fn=inside_mask)
    finally:
        rt.R0, rt.a0, rt.a1, rt.N_harm = R0_old, a0_old, a1_old, Nh_old

def build_torus_family(params_list: List[Dict], n_theta: int, n_phi: int) -> SurfacesDataset:
    items = []
    for i, p in enumerate(params_list):
        item = build_torus_surface_item(
            name=p.get("name", f"torus_{i}"),
            R0=float(p.get("R0", 1.0)),
            a0=float(p["a0"]),
            a1=float(p["a1"]),
            N_harm=int(p["N_harm"]),
            n_theta=n_theta, n_phi=n_phi
        )
        items.append(item)
    return SurfacesDataset(items)
