# streamlit run glass_batch_gui_inputs_v4.py
# GUI: 2–5 compounds glass batch calculator + MD-style box with per-component colours
import math
from typing import Literal, List, Dict
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

st.set_page_config(page_title="Glass Batch + MD Box (coloured)", page_icon="🧪", layout="wide")

Basis = Literal["mol%", "wt%"]

def calculate_masses(df_in: pd.DataFrame, total_mass_g: float, basis: Basis = "mol%") -> pd.DataFrame:
    if total_mass_g <= 0:
        raise ValueError("Total mass must be > 0 g")
    req = {"name","molar_mass_g_mol","fraction"}
    if not req.issubset(df_in.columns):
        raise ValueError(f"Data must contain: {sorted(req)}")

    df = df_in.copy()
    df["name"] = df["name"].astype(str).fillna("")
    df["molar_mass_g_mol"] = pd.to_numeric(df["molar_mass_g_mol"], errors="coerce")
    df["fraction"] = pd.to_numeric(df["fraction"], errors="coerce")
    if df.isna().any().any():
        raise ValueError("Please fill all fields with valid numbers.")

    frac_sum = df["fraction"].sum()
    if frac_sum <= 0:
        raise ValueError("Fractions must sum to a positive number.")
    if abs(frac_sum-100.0) > 1e-6:
        df["fraction"] = df["fraction"] * 100.0 / frac_sum

    if basis == "wt%":
        df["mass_g"] = total_mass_g * (df["fraction"]/100.0)
        df["moles_in_batch (mol)"] = df["mass_g"] / df["molar_mass_g_mol"]
    elif basis == "mol%":
        x = df["fraction"]/100.0
        M = df["molar_mass_g_mol"]
        denom = (x*M).sum()
        df["mass_g"] = total_mass_g * (x*M) / denom
        df["moles_in_batch (mol)"] = df["mass_g"] / df["molar_mass_g_mol"]
    else:
        raise ValueError("Unknown basis")

    df = df[["name","molar_mass_g_mol","fraction","mass_g","moles_in_batch (mol)"]]
    df = df.rename(columns={"fraction": f"{basis} (normalised)"})
    return df

st.title("🧪 Glass Batch Calculator — coloured MD-style Box (2–5 components)")

# Session state init
if "n" not in st.session_state:
    st.session_state.n = 3
if "rows" not in st.session_state:
    st.session_state.rows = [
        {"name":"Bi2O3","molar_mass_g_mol":465.96,"fraction":20.0},
        {"name":"TiO2","molar_mass_g_mol":79.87,"fraction":10.0},
        {"name":"TeO2","molar_mass_g_mol":159.60,"fraction":70.0},
        {"name":"","molar_mass_g_mol":0.0,"fraction":0.0},
        {"name":"","molar_mass_g_mol":0.0,"fraction":0.0},
    ]

with st.sidebar:
    st.markdown("### Settings")
    basis = st.radio("Composition basis", ["mol%","wt%"], index=0, horizontal=True)
    total_mass_g = st.number_input("Total batch mass (g)", min_value=0.0001, value=6.0, step=0.1, format="%.4f")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Add component", use_container_width=True):
            if st.session_state.n < 5:
                st.session_state.n += 1
            else:
                st.warning("Maximum is 5 components.")
    with c2:
        if st.button("Remove last", use_container_width=True):
            if st.session_state.n > 2:
                st.session_state.n -= 1
            else:
                st.warning("Minimum is 2 components.")
    if st.button("Reset example"):
        st.session_state.n = 3
        st.session_state.rows = [
            {"name":"Bi2O3","molar_mass_g_mol":465.96,"fraction":20.0},
            {"name":"TiO2","molar_mass_g_mol":79.87,"fraction":10.0},
            {"name":"TeO2","molar_mass_g_mol":159.60,"fraction":70.0},
            {"name":"","molar_mass_g_mol":0.0,"fraction":0.0},
            {"name":"","molar_mass_g_mol":0.0,"fraction":0.0},
        ]

st.markdown("### Enter components")
cols = st.columns([2,2,2])
cols[0].markdown("**Name**")
cols[1].markdown("**Molar mass (g/mol)**")
cols[2].markdown("**Fraction (% in chosen basis)**")

for i in range(st.session_state.n):
    c0, c1, c2 = st.columns([2,2,2])
    row = st.session_state.rows[i]
    row["name"] = c0.text_input(f"name_{i}", value=row["name"], label_visibility="collapsed", placeholder="e.g., Bi2O3")
    row["molar_mass_g_mol"] = c1.number_input(f"mm_{i}", value=float(row["molar_mass_g_mol"]), min_value=0.0, step=0.01, format="%.4f", label_visibility="collapsed")
    row["fraction"] = c2.number_input(f"frac_{i}", value=float(row["fraction"]), min_value=0.0, step=0.01, format="%.4f", label_visibility="collapsed")
    st.session_state.rows[i] = row

df_in = pd.DataFrame(st.session_state.rows[:st.session_state.n])

st.markdown("---")
st.write(f"Sum of fractions: **{df_in['fraction'].sum():.4f}%** (auto-normalised to 100%)")

st.markdown("### Results")
result = None
if not df_in.empty and df_in["fraction"].sum() > 0 and st.session_state.n >= 2:
    try:
        result = calculate_masses(df_in, total_mass_g=total_mass_g, basis=basis)
        st.dataframe(result, use_container_width=True)
        st.metric("Total mass (g)", f"{result['mass_g'].sum():.6f}")
        st.metric("Components", f"{len(result)}")
        st.download_button("Download results CSV", result.to_csv(index=False).encode("utf-8"),
                           file_name="glass_batch_result.csv", mime="text/csv")
    except Exception as e:
        st.error(str(e))
else:
    st.info("Enter at least two components and make sure the fraction sum is > 0%.")

# ---- MD-style box ----
st.markdown("---")
st.header("📦 MD-style Simulation Box (coloured per component)")
st.caption("Random placement visual by composition. Not a physical packing/dynamics simulation.")

with st.expander("Box settings", expanded=True):
    c1, c2, c3 = st.columns(3)
    box_len = c1.number_input("Box length (Å)", min_value=10.0, value=40.0, step=1.0)
    total_particles = int(c2.number_input("Total particles", min_value=100, value=2000, step=100))
    seed = int(c3.number_input("Random seed", min_value=0, value=42, step=1))

def _counts_from_fractions(fracs: np.ndarray, total_particles: int) -> List[int]:
    raw = fracs * total_particles
    base = np.floor(raw).astype(int)
    deficit = total_particles - base.sum()
    residuals = raw - base
    order = np.argsort(residuals)[::-1]
    for i in range(deficit):
        base[order[i % len(base)]] += 1
    base = np.where((fracs > 0) & (base == 0), 1, base)
    while base.sum() > total_particles:
        j = int(np.argmax(base))
        base[j] -= 1
    return base.tolist()

def _generate_positions(counts: List[int], L: float, seed: int):
    rng = np.random.default_rng(seed)
    pos = []
    types = []
    for t, count in enumerate(counts, start=0):
        if count <= 0:
            continue
        xyz = rng.random((count, 3)) * L
        pos.append(xyz)
        types.append(np.full(count, t, dtype=int))
    if not pos:
        return np.empty((0,3)), np.empty((0,), dtype=int)
    return np.vstack(pos), np.concatenate(types)

if result is not None:
    names = result["name"].tolist()
    frac_col = result.columns[2]  # the normalised fraction column
    fracs = result[frac_col].to_numpy() / 100.0

    counts = _counts_from_fractions(fracs, total_particles)
    pos, types = _generate_positions(counts, box_len, seed)

    # Colour map with up to 10 distinct colours
    cmap = plt.get_cmap("tab10")
    colours = [cmap(i) for i in range(len(names))]

    # Plot each type separately for legend
    fig = plt.figure(figsize=(6.5,6.5))
    ax = fig.add_subplot(111, projection="3d")
    for t, name in enumerate(names):
        mask = (types == t)
        if not np.any(mask):
            continue
        ax.scatter(pos[mask,0], pos[mask,1], pos[mask,2], s=2, color=colours[t], label=name)
    ax.set_xlim(0, box_len); ax.set_ylim(0, box_len); ax.set_zlim(0, box_len)
    ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)"); ax.set_zlabel("z (Å)")
    ax.set_title("Random placement by composition (coloured by component)")
    ax.legend(loc="upper right", fontsize=8, markerscale=4)
    st.pyplot(fig)

    # Export XYZ
    xyz_path = "sim_box.xyz"
    expanded_names = []
    for name, count in zip(names, counts):
        expanded_names += [name if name else "X"] * int(count)
    with open(xyz_path, "w") as f:
        f.write(f"{len(expanded_names)}\n")
        f.write("Generated by glass_batch_gui_inputs_v4\n")
        for (nm, (x,y,z)) in zip(expanded_names, pos):
            f.write(f"{nm} {x:.6f} {y:.6f} {z:.6f}\n")
    with open(xyz_path, "rb") as fh:
        st.download_button("Download XYZ (sim_box.xyz)", fh.read(), file_name="sim_box.xyz", mime="chemical/x-xyz")

    # Table of counts
    counts_df = pd.DataFrame({"name": names, "count": counts, "fraction_%": (np.array(counts)/sum(counts))*100})
    st.dataframe(counts_df, use_container_width=True)
else:
    st.info("Enter valid composition to enable box generation.")
