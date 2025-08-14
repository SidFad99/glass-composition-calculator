
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


st.set_page_config(page_title="Glass Composition Calculator & MD-style Box", layout="wide")


# ---------------------- ORIGINAL MASS CALC (verbatim logic) ----------------------
def calculate_masses(df_in: pd.DataFrame, total_mass_g: float, basis: str = "mol%") -> pd.DataFrame:
    """Follow the same rules as your original script:
       - Auto-normalise fractions to 100%
       - wt%: mass_i = total_mass * (fraction_i / 100)
       - mol%: mass_i = total_mass * (x_i * M_i) / sum_j (x_j * M_j)
       - moles_i = mass_i / M_i
    """
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
    if abs(frac_sum-100.0) > 1e-9:
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


# ---------------------- Helpers for box & counts ----------------------
def counts_from_fractions(fracs: np.ndarray, total_particles: int) -> np.ndarray:
    fracs = np.array(fracs, dtype=float)
    raw = fracs * total_particles
    base = np.floor(raw).astype(int)
    deficit = total_particles - base.sum()
    order = np.argsort(raw - base)[::-1]
    for i in range(deficit):
        base[order[i % len(base)]] += 1
    base = np.where((fracs > 0) & (base == 0), 1, base)
    while base.sum() > total_particles:
        j = int(np.argmax(base))
        base[j] -= 1
    return base.astype(int)


def box_length_from_density(total_mass_g: float, density_g_cm3: float) -> float:
    if density_g_cm3 <= 0:
        raise ValueError("Density must be positive.")
    # 1 cm = 1e8 Å → 1 cm^3 = 1e24 Å^3
    volume_A3 = (float(total_mass_g) / float(density_g_cm3)) * 1e24
    return float(volume_A3 ** (1.0 / 3.0))


def make_xyz_lines(names, counts, coords, box_len_A: float):
    total = int(sum(counts))
    lines = [str(total), f'Lattice="{box_len_A:.6f} 0 0 0 {box_len_A:.6f} 0 0 0 {box_len_A:.6f}"']
    start = 0
    for idx, (nm, c) in enumerate(zip(names, counts), start=1):
        end = start + int(c)
        tag = f"X{idx}" if not str(nm).strip() else str(nm).strip().replace(" ", "_")
        for x, y, z in coords[start:end]:
            lines.append(f"{tag} {x:.6f} {y:.6f} {z:.6f}")
        start = end
    return lines


# ---------------------- UI ----------------------
st.title("Glass Composition Calculator & MD-style Box")
st.caption("initiative by Siddiq Fadhil, UKM Malaysia")
st.caption("Mass per compound now follows the original calculation rules exactly.")

left, right = st.columns((1.2, 1))

with left:
    st.subheader("Composition")
    basis = st.radio("Input mode", ["mol%", "wt%"], horizontal=True)
    total_mass_g = st.number_input("Total batch mass (g)", min_value=0.0001, value=6.0, step=0.1, format="%.4f")

    st.caption("Enter 2–5 rows. Fractions are auto-normalised to 100%.")
    # Minimal table editor
    df_in = st.data_editor(
        pd.DataFrame([
            {"name":"Bi2O3","molar_mass_g_mol":465.96,"fraction":20.0},
            {"name":"TiO2","molar_mass_g_mol":79.87,"fraction":10.0},
            {"name":"TeO2","molar_mass_g_mol":159.60,"fraction":70.0},
        ]),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Compound"),
            "molar_mass_g_mol": st.column_config.NumberColumn("Molar mass (g/mol)", min_value=0.0, step=0.01, format="%.4f"),
            "fraction": st.column_config.NumberColumn("Fraction (%)", min_value=0.0, step=0.01, format="%.4f"),
        },
        hide_index=True,
        key="comp_table",
    )

    st.markdown("---")
    st.write(f"Sum of fractions: **{pd.to_numeric(df_in['fraction'], errors='coerce').fillna(0).sum():.4f}%** (auto-normalised to 100%)")

    st.subheader("Results")
    result = None
    try:
        result = calculate_masses(df_in, total_mass_g=total_mass_g, basis=basis)
        st.dataframe(result, use_container_width=True)
        st.download_button("Download results CSV", result.to_csv(index=False).encode("utf-8"),
                           file_name="glass_batch_result.csv", mime="text/csv")
    except Exception as e:
        st.error(str(e))


with right:
    st.subheader("MD-style Simulation Box (coloured per component)")
    st.caption("Random placement visual by composition. Not a physical packing/dynamics simulation.")

    with st.expander("Box settings", expanded=True):
        c1, c2, c3 = st.columns(3)
        target_density = c1.number_input("Target bulk density (g/cm³)", min_value=0.1, max_value=20.0, value=4.2, step=0.1)
        try:
            box_len = box_length_from_density(total_mass_g, target_density)
        except Exception as e:
            st.warning(f"Density → box length error: {e}")
            box_len = 40.0
        c1.number_input("Box length (Å)", value=float(box_len), disabled=True)
        total_particles = int(c2.number_input("Total particles", min_value=100, value=2000, step=100))
        seed = int(c3.number_input("Random seed", min_value=0, value=42, step=1))
        implied_density = (total_mass_g / ((box_len**3) / 1e24)) if box_len > 0 else 0.0
        st.caption(f"Implied density = {implied_density:.4f} g/cm³ (target {target_density:.4f}; Δ = {abs(implied_density-target_density):.4f})")

    if result is not None:
        names = result["name"].tolist()
        frac_col = result.columns[2]  # "<basis> (normalised)"
        fracs = result[frac_col].to_numpy() / 100.0

        counts = counts_from_fractions(fracs, total_particles)
        rng = np.random.default_rng(seed)
        coords = rng.random((int(sum(counts)), 3)) * float(box_len)

        # Plot
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        start = 0
        markers = ["o", "^", "s", "D", "P", "X", "*"]
        for i, (nm, c) in enumerate(zip(names, counts)):
            end = start + int(c)
            if c > 0:
                pts = coords[start:end]
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=6, alpha=0.8, marker=markers[i % len(markers)],
                           label=nm if str(nm).strip() else f"Comp-{i+1}")
            start = end
        ax.set_xlim(0, box_len); ax.set_ylim(0, box_len); ax.set_zlim(0, box_len)
        ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)"); ax.set_zlabel("z (Å)")
        ax.set_title("Random placement by composition (coloured by component)")
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0), fontsize=8)
        st.pyplot(fig, clear_figure=True)

        # XYZ
        xyz_lines = ["{}".format(int(sum(counts))), "Generated by app"]
        # expand names to match counts
        expanded_names = []
        for name, count in zip(names, counts):
            expanded_names += [name if str(name).strip() else "X"] * int(count)
        start = 0
        for nm, (x, y, z) in zip(expanded_names, coords):
            xyz_lines.append(f"{nm} {x:.6f} {y:.6f} {z:.6f}")
        st.download_button("Download XYZ", ("\n".join(xyz_lines)).encode("utf-8"),
                           file_name="sim_box.xyz", mime="chemical/x-xyz")
    else:
        st.info("Enter a valid composition above to enable the MD box.")
