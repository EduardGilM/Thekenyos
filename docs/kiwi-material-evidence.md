# Kiwi physical properties: evidence for simulation

Research checked 18 September 2026. This is a parameter-selection document, not a claim that the current simulator is calibrated. Implemented approximations are identified below; these are not joint calibration of the simulator.

**Use cultivar- and condition-specific presets.** Harvest-ready fruit is not necessarily soft eating-ripe fruit. Geometry, stiffness, damage and stem detachment cannot be sampled independently without creating unrealistic fruit. Keep the source, cultivar, temperature, firmness, loading rate and contact geometry with each preset.

Evidence labels below: **measured** = experimental observation; **fitted** = model parameter inferred from a particular experiment; **assumed** = modelling choice; **unknown** = no verified transferable range found. Reported ± values are reproduced as reported, not converted into population bounds.

## 1. Size, mass and density

| Source and population | Measured values | Application and limits |
|---|---|---|
| [Mu et al., 2020](https://doi.org/10.1016/j.inpa.2019.05.004), §2.1 / Table 1, Hayward, October harvests, 120 fruit | Mass **81.40–128.70 g**, mean **97.40 g**. Equatorial minor diameter **42.70–51.84 mm**, major **45.85–57.13 mm**. Mean axial length **64.98 mm**. | Useful harvest geometry prior. Table calls the two equatorial diameters minor/major axes; neither is axial fruit length. Text gives mean width52.16mm; table52.26mm, a small source discrepancy. Temperature/firmness not specified in the extracted methods. |
| [Razavi & BahramParvar, 2007](https://doi.org/10.2202/1556-3758.1276), abstract, Hayward | Length **55.5–82.3 mm**, width **46.8–54.8 mm**, thickness **41.5–52.4 mm**; mass **75.18–135.32 g**; true density **940–1040 kg/m³**; displaced volume **85–120 cm³**. | Broader shape/density bounds from another population; do not combine all independent extremes. Bulk density544.73–572.17kg/m³ includes gaps and must not become fruit material density. Harvest firmness, temperature and full sample protocol were not verified from accessible abstract. |
| [Urbańska et al., 2025](https://doi.org/10.1016/j.postharvbio.2025.113682), §3.1, SunGold | Three selected commercial size bands: **108–118**, **128–138**, **151–180 g**. | Selected export classes, not a uniform natural size distribution; do not label these Hayward. |

For a rigid approximation, compute inertia from actual geometry and mass. A solid ellipsoid with semi-axes a,b,c has Ixx=m(b²+c²)/5, and cyclic permutations. Derive density=m/mesh-volume, then check it against the selected population. Do not simultaneously prescribe conflicting mass, density and dimensions. A scanned fruit shape is preferable to a capsule for rolling and contact area.

## 2. Tissue elasticity and failure

These are tissue or fitted contact properties, not an actuator force limit. **1 MPa = 1,000,000 Pa.** 'Bio-yield' indicates the start of irreversible tissue deformation; it does not mean visible skin rupture. The papers use different failure definitions.

### Xuxiang static specimen dataset

[Zhu et al., 2024, Foods13:785](https://doi.org/10.3390/foods13050785), §§2.1–2.4, Table1. Fresh ripe Xuxiang; stored25°C/80%RH. Tissue loading3mm/min. Flesh/core cylindrical specimens diameter10±0.5mm, height12±0.5mm; skin strips50×15mm, thickness0.50±0.05mm.

| Tissue | Young modulus MPa | Post-yield tangent MPa | Bio-yield/failure MPa | Poisson ratio used |
|---|---:|---:|---:|---:|
| Skin | 10.233 | Not supplied | 0.514 | 0.30 |
| Core | 4.499 | 1.381 | 1.306 | 0.30 |
| Flesh, axial | 2.305 | 0.967 | 0.491 | 0.40 |
| Flesh, radial | 1.346 | 0.642 | 0.292 | 0.40 |

The uniform0.5mm skin shell is a model simplification. Poisson ratios are model inputs, not verified measurement distributions. Whole fruit was compressed at2.5%,5%,10%,20%; damage developed above5% in this experiment, assessed after16days at24°C/63%RH. This is not a universal strain threshold. Density entries have a unit error ('g/mm³') and inconsistent skin values between text and table: **do not import them**.

### Independent Xuxiang impact dataset

[Zhu et al., 2024, Foods13:3523](https://doi.org/10.3390/foods13213523), §§2.1–2.2 / Table3. Similar-maturity Xuxiang; tissue testing3mm/min,250mm plate,0.5N trigger. These are reported mean±variation, not hard ranges.

| Tissue | Young modulus MPa | Tangent modulus MPa | Bio-yield MPa | Density kg/m³ | Poisson ratio used |
|---|---:|---:|---:|---:|---:|
| Skin | 10.69±0.46 | — | 0.53±0.12 | 960±30 | 0.30 |
| Flesh | 1.57±0.12 | 0.92±0.06 | 0.26±0.07 | 1030±45 | 0.40 |
| Core | 5.11±0.28 | 0.83±0.04 | 1.12±0.23 | 1120±70 | 0.40 |

The paper examines0.25,0.5,1m drops, steel/PVC/neoprene, three orientations. Validation used0.5m onto steel at0°. Do not use its Table1 volume195,931.16mm³: it exceeds its64.18×53.46×50.72mm bounding box. Obtain mass/volume independently. No full operating-temperature or firmness distribution was verified.

### Grasping experiment cross-check

[Li et al., 2023](https://doi.org/10.3390/pr11020598), §§2.1–2.4 / Table1, uses **Xu Xiang**, despite a later paper describing it as Hayward. Tissue loading1mm/min;20 specimens per tissue. Model dimensions52×64×48mm.

| Tissue | Young modulus MPa | Breaking stress MPa | Density kg/m³ | Poisson ratio used |
|---|---:|---:|---:|---:|
| Peel | 11.20±0.50 | 1.51±0.21 | 551±50 | 0.30 |
| Flesh | 2.22±0.30 | 0.45±0.08 | 1122±20 | 0.30 |
| Core/placenta | 4.17±0.20 | 1.60±0.15 | 1063±40 | 0.30 |

Its specific plate-grasp test found little effect at0–5N, damage by25N, cracking above180N. These are **test-specific observations**, not safe Spot gripping forces. Packing through a900mm-height pipe caused damage, checked after72h. Peel density and failure criteria differ substantially from other studies. Do not average these datasets blindly.

### SunGold firmness-dependent whole-fruit response

[Urbańska et al., 2025](https://doi.org/10.1016/j.postharvbio.2025.113682), §§3.1–3.2 / Table2 ([open PDF](https://mro.massey.ac.nz/server/api/core/bitstreams/2010a7b0-6a47-46c1-9fbb-75d069326276/content)). Stored1°C/~98%RH for2–26weeks, measured20°C. Whole-fruit0.7N indentation,25mm-diameter glass ball, assumedν=0.3. The modulus is a **Hertz fit**, not an isolated-flesh coupon measurement.

| Cluster firmness N | Water loss % | Fitted Young modulus MPa at0.01mm/s |
|---:|---:|---:|
| 65 | 0.7 | 2.7 |
| 42 | 0.8 | 2.4 |
| 27 | 1.2 | 1.9 |
| 13 | 1.6 | 1.1 |
| 11 | 3.5 | 0.8 |
| 7 | 6.3 | 0.4 |

Firmness uses a7.9mm puncture probe at8mm/s after skin removal. It is not gripper force. The observed overall modulus changes roughly3→0.3MPa with storage. Rate tests0.01–0.2mm/s showed substantial rate dependence. These cluster centroids are coupled examples, not probability distributions or harvest-only ranges. The article describes probe radius inconsistently with its diameter; verify raw geometry before reproducing its Hertz calculation.

## 3. Time dependence: creep and relaxation

An elastic body that fully recovers after a23% squeeze does not demonstrate that a real kiwi would survive. Numerical damping is not a validated bruise model.

[Xie et al., 2023](https://doi.org/10.1016/j.foodp.2023.100005), §2.3 / Table4 ([author full text](https://www.researchgate.net/publication/375101247_Prediction_of_Physicochemical_and_Textural_Properties_of_Post-harvest_%27Hayward%27_Kiwifruit_Based_on_Creep_Properties)): Hayward~120g across storage, conditioned20°C; whole-fruit equatorial loading11N,1mm/s,30s. Four-element Burgers fit to **displacement**, with modelling-group ranges:

| Parameter | Reported range | SI conversion |
|---|---:|---:|
| Instantaneous stiffness E1 | 27.78–1432.93 N/mm | 27,780–1,432,930 N/m |
| Delayed stiffness E2 | 2.11–13.42 N/mm | 2,110–13,420 N/m |
| Viscous coefficient η1 | 103.13–2112.01 N·s/mm | 103,130–2,112,010 N·s/m |
| Viscous coefficient η2 | 0.42–17.67 N·s/mm | 420–17,670 N·s/m |

These correlated structural fits are **not Young moduli or MuJoCo damping coefficients**. Use the original geometry/load protocol to reproduce force–displacement–time response before transferring. Do not sample each extreme independently. The accessible methods did not establish a harvest-only subset.

[Lu et al., 2019](https://www.chinaagrisci.com/EN/10.3864/j.issn.0578-1752.2019.14.013), Table5, 'Haiwode' storage-series generalized Maxwell fit: modelling-group relaxation times **T1=0.201–23.696s, T2=0.188–25.060s, T3=0.152–23.918s**. Full protocol was not verified here. Useful evidence that relevant relaxation spans fractions of seconds to tens of seconds, not ready-to-copy material constants. Three branches are not three independent uniformly sampled delays.

## 4. Friction, impact and basket contacts

| Contact | Evidence | Use |
|---|---|---|
| Hayward–rubber, static | **0.38–0.51**, mean0.44; Mu2020§2.1, tilted rubber plate,10groups×3repeats | Suitable provisional range only for a comparable rubber and surface condition. |
| Kiwi–rubber/silicone/nylon, static | Mean **1.0799 / 0.5520 / 0.4663**; [Dhanotra2024 thesis](https://hdl.handle.net/10289/17431), Table3.4,p35, inclined surfaces | Different material specimens produce very different friction. Cultivar, wetness and pad formulation not verified from available extract. Do not turn three means into a universal range. |
| Hayward–glass/plywood | **0.34 / 0.49**; Razavi2007 abstract | Surface-specific static values; neither is basket plastic. |
| Kiwi–kiwi restitution | [Wu2022](https://doi.org/10.1016/j.jfoodeng.2022.111060), Hayward85–100g, days0/4/8: changes with speed, ripeness and posture. No bruises observed where measured CoR>0.58. | Full coefficient-vs-speed table not accessible. **Unknown default.** Setting e>0.58 does not make an impact safe. |
| Kiwi–kiwi sliding/rolling friction | No directly verified numerical range found | Measure; keep as calibration-needed. |
| Kiwi–actual basket/liner, wet/dry friction and restitution | No directly verified numerical range found | Test the chosen liner and shell. Do not substitute stem–steel coefficients. |

[Wu's2021 primary conference experiment](https://doi.org/10.13031/aim.202100104) reports kiwi–kiwi CoR>0.58 at the lowest tested1.1m/s collision speed; variability increases with ripening. This is a validation point, not a complete collision law. Record relative normal/tangential velocities, orientation, contact impulse, absorbed energy and repeat impacts for each fruit. A ground drop can be a task failure immediately, independently of whether a calibrated tissue model predicts bruising.

[The2025 repeated-impact study](https://doi.org/10.3389/fpls.2025.1683638) uses harvest-stage Xuxiang,1/3/5impacts at three locations, followed42days at4°C/72%RH. It supports storing cumulative damage history and inspecting delayed quality loss; it does not supply a general safe impulse threshold.

## 5. Stem and detachment

| Parameter | Evidence / range | Limit |
|---|---|---|
| Hayward stalk diameter/length | Mean **3.68mm / 43.84mm**,100stalks,2022harvest; moisture39.5–46%,tested within24h | Means only; no verified numeric min/max from the distribution figure. |
| Stalk density / Young modulus / Poisson ratio | **867.5kg/m³ / 325MPa / 0.26** | Tensile test2mm/min,20mm mid-stem specimen; not abscission-joint properties. |
| Stalk three-point bending | Force **3.23–4.52N**, strength **5.52–7.99MPa** across20specimens; mean3.89N/6.75MPa |2mm/s; requires original support span before reproducing the experiment. Not fruit detachment. |
| Stem–steel restitution | **0.365** | Stem only; not kiwi fruit restitution. |

Above: [He et al.,2024](https://doi.org/10.4081/jae.2024.1640), §§Materials/Methods,Table1–2 ([PDF](https://www.agroengineering.org/jae/article/download/1640/1254/11312)). The abstract and Table2 disagree on friction labels/values; standard-deviation entries also appear dimensionally suspect. Exclude those friction/uncertainty fields. Its calibrated particle-bond stiffness and failure stress depend on DEM particle/bond sizes and must not be copied into a continuum stem or a breakable MuJoCo joint.

For detachment itself, Mu2020Table1 reports **1.08–12.25N**, mean4.91N, pooled over angle tests; mean stalk length58.7mm in that separate sample. Its minimum force occurred near60° between fruit and stem axes. **This pooled range is not an angle-conditioned break law.**

[Fang et al.,2023](https://doi.org/10.1016/j.compag.2023.108225) tests60–180° in five cultivars. Minimum detachment occurred at60° for Hayward/Xuxiang/Huayou,80° for Qinmei/Cuixiang; Xuxiang stems could break at160/180°. The user-supplied full paper is now available. The tests used 210 fruit (42 per
cultivar), six fruit per angle, a 9 mm/s fixture pull, and fruit tested within
four hours of harvest on 23 October 2022 in Zhouzhi, China. Fruit–stem angle is
between the junction-to-fruit-tip vector and the junction-to-stem-anchor vector:
a straight hanging fruit is 180 degrees.

The implemented Hayward mean-force proxy uses the following points:

| Angle degrees | Mean detachment force N | Evidence |
|---:|---:|---|
| 60 | 5.98 | Reported in text |
| 80 | 6.3 | Approximate visual reading of Fig. 7 mean marker |
| 100 | 13.8 | Approximate visual reading of Fig. 7 mean marker |
| 120 | 21.3 | Approximate visual reading of Fig. 7 mean marker |
| 140 | 30.7 | Approximate visual reading of Fig. 7 mean marker |
| 160 | 40.27 | Reported in text |
| 180 | 36.5 | Approximate visual reading of Fig. 7 mean marker |

Intermediate readings have roughly 1 N digitization precision; their decimals
are not raw experimental precision. Linear interpolation and clamping below
60 degrees are **assumptions**. Only tensile load along the instantaneous stem
axis triggers this rule. The default strength multiplier is one; changing it
is an engineering sensitivity test, not a measured population distribution.
The mean curve does not reproduce experimental variance, rate dependence,
shear failure, torque failure or cultivar differences. This replaces the older
Mu2020 pooled uniform-force sampling. Geometry, stem stiffness and detachment
still come from different experimental populations.

These forces are attachment loads, not safe jaw forces. The paper's preferred
fixture angle does not prescribe the optimal motion for Spot's gripper. Verified kiwi-specific abscission torsional stiffness, failure torque, and fracture-energy ranges remain **unknown**. Cutting torque is not twisting-detachment torque.

For a reduced model, use a bending/torsion-capable stalk connected at the fruit's actual stem site, with a separate breakable attachment. Derive beam EA/EI from stem geometry and tissue modulus if used; calibrate the break law separately. A COM spring cannot represent the correct moment arm or picking-angle response.

## 6. Initial simulation choices and measurements still needed

These are **engineering proposals**, not measured distributions:

1. **Free fruit:** start with the Mu Hayward geometry/mass envelope. A uniform bounded sampler is acceptable for robustness testing, but call it domain randomization. Couple mass and volume; reject inconsistent density. Give every detached/basket fruit its own mass, inertia, translation, rotation and collision geometry. No fixed ballast or visual-only fruit in a spill evaluation.
2. **Contacts:** useμs∈[0.38,0.51] only for a provisional comparable rubber-gripper case. Leave plastic/liner, fruit–fruit, kinetic friction and rolling resistance explicitly uncalibrated. Resolve them with tilt/slide/rolling/drop tests. MuJoCo solver softness is a numerical parameter and needs contact-response fitting.
3. **Deformation:** reproduce one complete source experiment first. A layered Xuxiang preset can start from the impact table; a SunGold surrogate should fit one coupled firmness/water-loss cluster. Neither should be called calibrated Hayward. The existing30kPa toy modulus is not an empirical harvest-kiwi value.
4. **Damage:** retain separate irreversible bruise, peel rupture and stem failure state. Train against calibrated damage and spill events, not only appearance or peak gripper force. Before calibration, report a 'damage proxy', never 'no bruising'. Contact pressure requires a resolved/fitted contact patch; force alone is insufficient.
5. **Calibration campaign:** sample the intended cultivar at harvest; record temperature, mass, three axes, displaced volume, firmness and moisture loss. Test compression/release at grasp speeds and holds; compare multiple locations and pad areas. Inspect tissue immediately and after storage. Fit parameters on one subset and evaluate the remaining fruit.
6. **Detachment campaign:** measure force/torque versus pulling direction and fruit angle, stem dimensions, detachment location and damage. Add twisting only after its torque response is measured. Preserve low-force bending behaviour without allowing spontaneous gravity detachment.
7. **Basket campaign:** drop individual fruit onto the actual liner and other fruit over a height/speed range, then run repeated shaking and spill tests. Match rebound, rolling, packing and delayed damage. Loss from the basket is independently observable and can already be penalized; calibrated bruise prediction comes later.

No research found here provides a complete, jointly validated Hayward-or-SunGold parameter distribution covering all these behaviours. The useful result is a set of measured starting envelopes, reproducible benchmark protocols, and explicit calibration gaps—not a claim that all constants can safely be copied into one material.
