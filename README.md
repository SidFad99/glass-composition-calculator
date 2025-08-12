
# Glass Batch Calculator + MD-style Box

A small Streamlit app to size glass batches from **mol%** or **wt%** and to create a simple **MD-style coloured box** that places particles in a cube with counts proportional to the normalised composition. The box is a visual aid, not a physics simulation.

## Features
- Inputs for **2–5** components on the page (name, molar mass g/mol, fraction %).
- Works with **mol%** or **wt%**; fractions are auto-normalised to 100%.
- Per‑component **mass (g)** and **moles (mol)** for a chosen total batch mass.
- 3D box view with **different colours per component** and an **XYZ export** for OVITO/VMD.

## Quick start (local)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open the local URL printed in the terminal (usually `http://localhost:8501`).

## Streamlit Community Cloud (free)
1. Push this folder to a **public GitHub repo**.
2. Go to **https://streamlit.io/cloud** and sign in.
3. Choose **New app** → select your repo → set **Main file path** to `app.py` → **Deploy**.
4. You’ll receive a public URL you can share.

Docs:
- Streamlit Cloud: https://docs.streamlit.io/streamlit-community-cloud
- GitHub: https://docs.github.com/en/get-started/quickstart/create-a-repo

## Using the app
1. Choose **mol%** or **wt%** in the sidebar and enter the **total mass (g)**.
2. Set the number of components (2–5) with **Add / Remove**.
3. Fill in **Name**, **Molar mass (g/mol)**, and **Fraction (%)** for each row.
4. Results table shows the batch mass for each compound and the moles present.
5. The **MD-style box** panel lets you set **box length (Å)**, **total particles**, and **random seed**. Download `sim_box.xyz` for visualisation.

### Notes on the MD-style box
- This is a **random placement** in a cube; no force field, no energy minimisation, and no overlaps are resolved. It is intended for teaching/demonstration.
- The XYZ contains one point per particle with the compound name as the element label.

## File tree
```
.
├── app.py
├── requirements.txt
└── README.md
```

## Licence
MIT Licence. See below.

---

MIT License

Copyright (c) 2025
... (licence text continues as standard MIT) ...
