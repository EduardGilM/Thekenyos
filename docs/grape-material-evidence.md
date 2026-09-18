# Grape cluster physical properties: evidence for the vineyard environment

Research checked 18 September 2026. This is a parameter-selection document for the
`vineyard` preset, not a claim that the grape simulation is calibrated. It follows
the same labelling as `kiwi-material-evidence.md`: **measured** = experimental
observation; **fitted** = model parameter inferred from a particular experiment;
**assumed** = modelling choice; **unknown** = no verified transferable range found.

Scope: whole table-grape clusters (bunches) hanging from an overhead arbor. The
simulation treats one cluster as one rigid body. Individual berries are visual
shapes without collision; berry-level deformation, berry drop and rachis
flexibility are not modelled.

## 1. Cluster size, mass and peduncle diameter

| Source and population | Measured values | Application and limits |
|---|---|---|
| [Hu, Zhang & Hu, 2025](https://doi.org/10.3390/agronomy15122813), Table 1, table grapes Kyoho / Rose / Red Globe | Cluster length **130–290 mm**; equatorial diameter **90–200 mm**; mass **300–2000 g**; pedicel (peduncle) diameter **4.6–10.8 mm**. Per-variety: Kyoho 140–245 mm, 100–200 mm, 300–1600 g, 5.8–10 mm; Rose 130–200 mm, 90–190 mm, 320–1300 g, 4.6–9.5 mm; Red Globe 180–290 mm, 100–180 mm, 400–2000 g, 5.3–10.8 mm. | Used as the sampling envelope for cluster geometry, mass and peduncle diameter in `treesim/grape_material.py`. Ranges are the paper's reported min–max over three varieties; the joint distribution (long clusters are heavier) is **not** reported, so the sampler couples them with a bulk-density rejection bound (assumed, section 4). Sample size, ripeness and temperature are not given in the extracted text. |

## 2. Berry and peduncle failure

| Source | Measured values | Application and limits |
|---|---|---|
| [Hu, Zhang & Hu, 2025](https://doi.org/10.3390/agronomy15122813), Table 2 | Single-berry critical rupture force, lateral compression **27.96–34.54 N**, longitudinal **25.79–29.51 N** (Kyoho / Rose / Red Globe). Pedicel shear force **85.37–92.36 N**; shear strength **1.349–1.426 MPa**. | The lowest value, **25.79 N**, is the berry-rupture reference used by the contact diagnostic. It is a single-berry plate test; a whole-cluster contact spreads over several berries, so exceeding it in the simulation flags *risk*, it does not predict rupture. The pedicel shear force is a blade-cutting load, **not** a tensile abscission force. |
| Same paper, text | Gripper contact force of 11 N per finger clamped a 2 kg cluster without damage in field trials (96.7 % success, 3.2 % cluster damage, 2.8 % berry drop). | Test-specific observation for one end-effector; not a safe gripping force for Spot. |

## 3. Unknowns kept as explicit calibration gaps

- **Tensile detachment force of the peduncle** (pull-off without cutting): unknown. Real table-grape harvest severs the peduncle. The simulation samples a tensile threshold uniformly in **40–90 N** (assumed), bounded above by the measured shear force and below by a margin above the heaviest cluster weight (2 kg -> 19.6 N). Use `GrapeField.cut(i)` to model harvesting; pull-off is a fallback proxy only.
- **Peduncle Young's modulus**: unknown for table grapes. Assumed **300 MPa**, the order of magnitude of a woody stem (compare He 2024 kiwi stem 325 MPa). This sets tether stiffness only; it does not change detachment strength.
- **Peduncle length** between shoot node and first rachis branch: unknown; assumed **40–60 mm**.
- **Berry diameter, count per cluster and cluster porosity**: not in the sources above. Visual berries use an assumed radius of **10–13 mm**; they carry no mass or collision. Cluster bulk density (section 4) stands in for porosity.
- **Cluster/gripper friction and restitution**: unknown; the code reuses the kiwi proxies (mu 0.6, restitution 0.05) and labels them as proxies.
- **Rachis compliance and berry shedding** under contact: not modelled. A rigid cluster overstates transmitted contact force and cannot drop berries.

## 4. Derived and assumed sampling rules (engineering)

Cluster collision geometry is a solid ellipsoid with semi-axes (D/2, D/2, L/2).
Mass, D and L are sampled from the measured envelope and accepted only when the
implied bulk density m / (4/3 pi (D/2)^2 (L/2)) lies in **250–700 kg/m^3**
(assumed: below the ~1000 kg/m^3 of berry tissue because of air gaps, above a
loose-fill floor). Inertia is computed by the physics engine from that geometry
and mass. This rejection rule is a robustness envelope, not a measured joint
distribution; do not report it as cluster density data.

The peduncle tether stiffness uses E A / L axially and 3 E I / L^3 in bending
from the sampled diameter and assumed E and L. As for kiwi, breaking is
triggered by the actual spring load, so contacts and canopy motion can detach
a cluster as well as a pull.
