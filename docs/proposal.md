# A power-balance atlas of BLI benefit, quantified under uncertainty

**Bottom line up front.** A single-researcher, purely computational project can produce a genuinely novel, publishable contribution to boundary-layer ingestion (BLI) propulsion research in 2025–2026 by delivering the **first open-source, pip-installable Python implementation of Drela's power-balance BLI benefit model unified with the Smith 1993 and Hall 2017 formulations, coupled to formal polynomial-chaos + Monte-Carlo uncertainty quantification and Sobol' global sensitivity analysis** across the joint design/uncertainty space. This fills two verified gaps: (i) no such reproducible open-source tool exists, and (ii) no peer-reviewed paper has applied rigorous UQ/GSA to a low-order BLI benefit model. Existing high-fidelity CFD-based BLI studies (Yildirim et al. 2022, Chau et al. 2024–2025) are physically superior but structurally incapable of exhaustive parametric coverage or of publishing reproducible artifacts. The proposal below is executable by one researcher in **~22 weeks at ~20 h/week** using only `numpy/scipy/matplotlib/SALib/chaospy`, and targets the *Journal of Propulsion and Power* (primary), with an arXiv preprint on submission day and a JOSS software companion. The dominant technical risks — thrust/drag bookkeeping ambiguity, control-volume consistency, and boundary-layer model validity on the tail cone — are all resolvable within the power-balance framework and are mitigated by concrete engineering practices detailed below.

---

## 1. Physics and mathematics foundation

### 1.1 Drela's power balance: what it is and why it exists

Drela's 2009 *AIAA Journal* paper "Power Balance in Aerodynamic Flows" was written specifically to sidestep an intractable problem: for BLI configurations, the airframe boundary layer and propulsive jet share a common downstream stream tube, and there is no unique way to separate "airframe drag" from "propulsor thrust." The traditional streamwise-force decomposition $F_u \equiv D_p - T$ becomes ambiguous, because the dividing streamline between "wake" and "jet" is a modeling choice, not a physical surface. **The power balance replaces this ambiguity with a positive-definite scalar accounting of mechanical energy sources and dissipations, in which no thrust and no drag ever appear.**

Starting from the compressible momentum equation dotted with the velocity vector, Drela integrates the resulting mechanical-energy equation over a control volume bounded by the vehicle surface $S_B$, a Trefftz plane $S_O^{TP}$ perpendicular to the freestream, and a lateral "side cylinder" $S_O^{SC}$. After application of Gauss's theorem and exact decomposition, the working form of the integral power balance is

$$P_S + P_V + P_K \;=\; W\dot h + \dot E_a + \dot E_v + \dot E_p + \dot E_w + \Phi. \tag{1}$$

Each term is a **scalar power**. On the input side: $P_S$ is the shaft/electric power delivered through solid propulsor surfaces inside the CV; $P_V$ is a volumetric $p\,\nabla\!\cdot\!\vec V$ term nonzero only if combustion occurs inside the CV; $P_K$ is the mechanical flow power carried across propulsor cut planes,

$$P_K = \oiint_{S_B}\!\bigl[(p-p_\infty)+\tfrac12\rho(V^{2}-V_\infty^{2})\bigr]\vec V\!\cdot\!\hat n\;dS_B. \tag{2}$$

On the output side: $W\dot h$ is the potential-energy climb rate; $\dot E_a$ is streamwise kinetic-energy outflow through the Trefftz plane (positive for both propulsive jets and viscous wakes); $\dot E_v$ is transverse (trailing-vortex) KE outflow; $\dot E_p$ is a wake pressure-defect work rate; $\dot E_w$ captures lateral wave losses through the side cylinder; and $\Phi$ is the total viscous and shock dissipation inside the CV,

$$\Phi = \Phi_{\text{surf}} + \Phi_{\text{wake}} + \Phi_{\text{jet}} + \Phi_{\text{vortex}} + \Phi_{\text{shock}}. \tag{3}$$

For a shear layer this local dissipation is $\Phi = \iint \rho_e u_e^{3}\,C_{\mathcal D}\,dx\,dz$ with dissipation coefficient $C_{\mathcal D}$. **Every entry in (1) is a positive-definite volume or surface integral of the actual physics; the framework contains no drag/thrust split by construction.** This is the single feature that makes the power balance the correct language for BLI benefit quantification.

### 1.2 Boundary-layer momentum and kinetic-energy deficits

Along a wall coordinate $x$, edge velocity $u_e$, and wall-normal $y$, define

$$\theta = \int_0^{y_e}\!\tfrac{u}{u_e}\!\bigl(1-\tfrac{u}{u_e}\bigr)dy, \quad \delta^{*} = \int_0^{y_e}\!\bigl(1-\tfrac{u}{u_e}\bigr)dy, \quad \theta^{*} = \int_0^{y_e}\!\tfrac{u}{u_e}\!\bigl[1-(\tfrac{u}{u_e})^{2}\bigr]dy. \tag{4}$$

Momentum thickness $\theta$ measures the momentum deficit and links to profile drag through the far-field wake integral. **Kinetic-energy thickness $\theta^{*}$ is the central quantity for the power-balance method**: neglecting the compressibility density-flux term, integration of the KE integral equation gives

$$\tfrac12\rho_e u_e^{3}\,\theta^{*}(x) \;=\; \int_0^{x}\!\rho_e u_e^{3}\,C_{\mathcal D}\,dx \;=\; \Phi(x). \tag{5}$$

**The kinetic-energy thickness at any station equals the cumulative upstream dissipation.** Physically, the boundary layer carries within it an integrated ledger of the energy that has already been irreversibly lost to viscous friction upstream. At the trailing edge, this KE deficit represents dissipation that is either (a) subsequently lost to wake mixing (in the podded reference case), or (b) potentially recoverable by an ingesting propulsor (in the BLI case). Introducing the wake KE-excess thickness $\delta_K = 2\theta - \theta^{*}$, the axial KE flux carried into the wake is $\dot E_a = \tfrac12\rho_e u_e^{3}\,\delta_K$ per unit span.

### 1.3 Why ingesting low-momentum flow reduces required power

For an isolated actuator disk producing thrust $F$ with mass flow $\dot m$, jet velocity $u_j$, and inflow $u_1 = u_\infty$, the required flow power is $P_{\text{no BLI}} = \tfrac{F}{2}(u_j+u_\infty)$. If the same $\dot m$ instead ingests a boundary-layer flow with mass-averaged velocity $u_w < u_\infty$, then for the same net force $F = \dot m(u_j' - u_w)$ the fan needs a lower jet velocity $u_j' < u_j$ and delivers less kinetic-energy addition:

$$P_{\text{BLI}} = \tfrac{F}{2}(u_j'+u_w) \;<\; P_{\text{no BLI}}. \tag{6}$$

Equivalently, in Drela's ledger, the podded case must dissipate the *full* fuselage wake ($\Phi_{\text{wake}}^{\text{full}}$) and pay a large jet-mixing loss ($\Phi_{\text{jet}}^{\text{pod}}$) because a high-velocity jet enters undisturbed freestream. The BLI case partially re-energizes the wake (reducing $\Phi_{\text{wake}}$) and produces a smaller velocity jump $\Delta u = u_j' - u_w$, which quadratically reduces $\Phi_{\text{jet}}$. The savings are physically real and traceable term-by-term in the power balance.

### 1.4 BLI metrics and closed-form estimators

The community-standard figure of merit is the **Power Saving Coefficient** (Smith 1993, retained by Uranga 2017/2018),

$$\text{PSC} \;=\; \frac{P_{\text{non-BLI}} - P_{\text{BLI}}}{P_{\text{non-BLI}}} \Bigg|_{\text{same } C_X, C_L}, \tag{7}$$

evaluated at equal net streamwise force coefficient $C_X = F_x/(\tfrac12\rho_\infty V_\infty^{2}S_{\text{ref}})$ (in cruise, $C_X = 0$) and equal lift coefficient. Non-dimensional flow-power coefficient: $C_{P_K} = P_K/(\tfrac12\rho_\infty V_\infty^{3}S_{\text{ref}})$.

Smith's 1993 wake-ingestion formula relates PSC to ingested wake fraction, thrust loading, and wake shape factor; combined with the Drela/Hall parametric decomposition (Hall, Huang, Uranga, Greitzer, Drela & Sato, *J. Propulsion & Power* 33(5), 2017), the compact working expression is

$$\text{PSC} \;\simeq\; f_{BLI}\;\eta_{\text{fill}}\;\frac{(V_j - V_w)}{(V_j + V_\infty)}\bigl[1 + \text{FPR-correction}\bigr], \tag{8}$$

with $f_{BLI}$ the dissipation-fraction ingested, $\eta_{\text{fill}} \in [0,1]$ the fill/attenuation factor (fraction of the ingested KE defect the fan actually re-energizes), and $V_w$ the mass-averaged wake velocity at the fan face. **The three physical levers are exactly $f_{BLI}$, $\eta_{\text{fill}}$, and disk loading (FPR).** For a fully-swallowing ideal actuator disk the benefit tends to $\Phi_{\text{jet}}^{\text{pod}}/P_K^{\text{pod}}$, i.e. the entire jet-mixing loss of the reference case is recovered.

### 1.5 Fuel burn and the turboelectric bookkeeping

Cruise fuel burn follows Breguet,

$$R = \frac{V_\infty}{g\,\text{TSFC}}\,\frac{L}{D}\,\ln\!\frac{W_i}{W_f}, \quad \eta_{\text{overall}} = \eta_{\text{thermal}}\,\eta_{\text{transmission}}\,\eta_{\text{propulsive}}. \tag{9}$$

To first order, at fixed range, $L/D$, and payload, $\Delta W_{\text{fuel}}/W_{\text{fuel}} \approx -\text{PSC}\cdot\phi$, where $\phi$ is the fraction of total propulsive power routed through the BLI propulsor (STARC-ABL: $\phi \approx 0.28$; D8: $\phi \approx 1.0$). A "snowball" resizing factor of ~1.2–1.5 amplifies this for rubber-airframe studies.

For a turboelectric BLI fan (STARC-ABL architecture: underwing turbofans → generator → cable/inverter → motor → aft fan), the electrical chain injects a loss

$$P_K^{\text{aft}} = \eta_g\,\eta_c\,\eta_m\,P_{\text{shaft, extracted}}, \quad \eta_{\text{elec}} \approx 0.90\text{–}0.93, \tag{10}$$

with NASA reference values $\eta_g \approx 0.96$, $\eta_c \approx 0.98$–$0.996$, $\eta_m \approx 0.96$. The **net BLI benefit** in the turboelectric case is

$$\text{PSC}_{\text{net}} = \text{PSC}_{\text{aero}} - (1 - \eta_{\text{elec}})\,\frac{P_K^{\text{aft}}}{P_{K,\text{total, non-BLI}}}. \tag{11}$$

For STARC-ABL this converts an ~11% aerodynamic PSC into a ~3–7% block-fuel benefit, matching the NASA TM-20210016661 numbers.

---

## 2. Reference configurations and the state of the literature

### 2.1 STARC-ABL: the benefit that shrank from 12% to 3%

NASA's Single-aisle Turboelectric AiRCraft with Aft Boundary-Layer propulsion is a ~154-passenger, tube-and-wing platform with two under-wing turbofans driving generators that feed a tail-cone electric BLI fan (aft FPR ≈ 1.25, absorbing ~1/3 of cruise thrust). Its published benefit has evolved dramatically. **The Welstead & Felder 2016 conceptual estimate of 12% design-mission fuel-burn reduction** (AIAA 2016-1027) is the number most often quoted but is now obsolete. After the Rev B M0.7→M0.785 revision and cleanup (Bowman/Felder 2018), the number dropped to **3.4% design-mission / 2.7% economic-mission**, canonicalized in NASA/TM-20210016661 (Felder et al. 2022). Coupled RANS + NPSS optimization by Yildirim, Gray, Mader, and Martins (*J. Aircraft* 59(4), 2022) reports an isolated-propulsor BLI benefit of only **2.1–2.3% power savings**. **Any modern paper must cite the updated numbers, not the 2016 conference number, or reviewers will flag it.** The consensus community range is now **~2–4% block-fuel reduction** for STARC-ABL, and coupled analysis is essential — uncoupled studies systematically overestimate the benefit.

### 2.2 D8 "double bubble": the reference experimental number

The MIT/Aurora D8 concept — 180 pax, M0.74, double-bubble lifting fuselage, twin aft-mounted podded BLI engines — is the most cited *experimental* BLI benefit in the literature. Uranga, Drela, Greitzer, Hall et al. (*AIAA Journal* 55(11), 2017) reported **8.2% ± 0.8% mechanical flow-power reduction** at cruise conditions from back-to-back BLI vs. non-BLI 1:11 wind-tunnel testing at NASA Langley 14×22-ft. Hall et al. 2017 decomposed this into 5.2% from jet dissipation reduction, 2.4% surface, 0.6% wake. Extrapolated to full-scale block fuel, MIT/Aurora attribute ~7–9% to BLI itself (the D8's total ~30–40% fuel-burn improvement is dominated by airframe design). **This is the low-Mach, tunnel-scale number, and the paper's validation must reproduce the equal-nozzle-area comparison rule Uranga used, not force a single global match.**

### 2.3 N3-X, propulsive fuselage, and other configurations

The NASA N3-X hybrid-wing-body with superconducting turboelectric distributed propulsion (Felder/Kim 2011, Kim et al. 2014) claims a 70–72% total mission energy reduction vs. B777-200LR, of which BLI+TeDP contribute ~18–20%. The Bauhaus Luftfahrt / DisPURSAL and CENTRELINE propulsive fuselage concepts (Isikveren/Seitz/Bijewitz 2015; Seitz et al. 2021) report 4.7% design-mission fuel benefit for optimized aeroshaping, in a 2.8–3.7% range across assumption sets. Giannakakis/Safran 2019–2020 sound the cautionary note: **when installed BLI-fan mass and electrical-chain penalties are honestly counted, aft-fuselage BLI can produce +1.7% ± 1.0% *increase* in fuel burn** for the propulsive-fuselage class. The 2025 Castillo-Pardo & Hall study of a six-tail-propulsor T-tail single-aisle at M0.8 reports 8.5% payload-range fuel benefit — the freshest reference for the review section. Ma, Li & Li's 2025 *Progress in Aerospace Sciences* review (154, 101082) and Moirou/Sanders/Laskaridis 2023 (138, 100897) are the two literature landmarks to anchor the survey.

**Consolidated benefit table** (subset; full table in the proposal appendix):

| Configuration | Source | Metric | Value |
|---|---|---|---|
| STARC-ABL Rev A | Welstead & Felder 2016 | Design-mission fuel | 12% (obsolete) |
| STARC-ABL Rev B | Felder et al. NASA/TM-20210016661 (2022) | Design-mission fuel | **3.4%** |
| STARC-ABL isolated propulsor | Yildirim et al. 2022 | Power savings (RANS+NPSS) | 2.1–2.3% |
| D8 wind tunnel | Uranga et al. 2017 | Mech. flow-power, cruise | **8.2% ± 0.8%** |
| D8 decomposition | Hall et al. 2017 | Jet+surface+wake | 5.2+2.4+0.6% |
| CENTRELINE PFC | Seitz et al. 2021 | Design-mission fuel | 4.7% (2.8–3.7% range) |
| Turboelectric PFC | Giannakakis et al. 2020 | Fuel burn Δ | +1.7% ± 1.0% (penalty) |
| N3-X | Kim/Felder 2014 | BLI+TeDP attributable | 18–20% |
| Aft 6-propulsor T-tail | Castillo-Pardo & Hall 2025 | Payload-range fuel | 8.5% |

**The credible BLI-only fuel-burn benefit envelope for tube-and-wing/PFC classes is 2–9%.** Higher numbers require unconventional airframes or optimistic technology assumptions.

### 2.4 Existing open-source Python landscape: the verified gap

A systematic search of GitHub, PyPI, and JOSS confirms that **no dedicated Drela power-balance BLI benefit calculator exists in open source**. NASA's Aviary (2024–2025), OpenConcept (Brelje/Adler/Martins), SUAVE/RCAIDE, and OpenAP do not include a first-principles power-balance BLI module — BLI can be entered as an aeropropulsive-coupling assumption but not decomposed via Drela's ledger. Gray/Yildirim/Martins's STARC-ABL pipeline depends on the proprietary ADflow RANS solver. **This is a real, unfilled gap in the reproducible-tooling landscape.**

### 2.5 The three ranked novelty gaps for 2025–2026

Ranked by defensibility, publishability, and 3–6-month feasibility:

**Rank 1 (strongest).** A pip-installable, JOSS-published Python implementation of the Drela/Hall power-balance BLI benefit model, with formal polynomial-chaos + Monte-Carlo UQ and Sobol' global sensitivity analysis over the joint space (BLI fraction, FPR, BL thickness, Mach, altitude, fan/duct loss, transmission efficiency, installed-mass penalty). *No such open-source tool exists; no peer-reviewed paper applies rigorous UQ/GSA to a low-order BLI benefit model.* Uranga 2017's ±0.8% is instrument-error propagation on tunnel data, not UQ on the analytical model. This yields a JOSS software paper plus a research paper on parameter importance, saturation, and probability distribution of block fuel — content CFD studies cannot afford.

**Rank 2.** A unified, apples-to-apples parametric comparison of Smith 1993, Drela 2009, and Hall 2017/Sabnis 2023 frameworks in one open-source pipeline, applied to STARC-ABL, D8, and a propulsive-fuselage archetype with the Habermann 2020 consistent bookkeeping, reconciling published benefit numbers. Explicitly called for by Ma et al. 2025 and Moirou et al. 2023 reviews.

**Rank 3.** A full 5-D parametric benefit atlas (BLI fraction × FPR × BL thickness × Mach × altitude) with explicit diminishing-returns / saturation structure. Yildirim 2022 and Giannakakis 2020 partially do this but not reproducibly and not with UQ.

**Combining Ranks 1 + 2 into one paper (with Rank 3 as its parametric atlas) is realistic, defensible, and non-overlapping with the CFD literature.**

---

## 3. Computational methodology

### 3.1 The integral boundary-layer chain: Thwaites → Head → wake

The core BL solver returns $(\theta, \delta^{*}, \theta^{*}, C_f, C_{\mathcal D})$ along a body with prescribed edge velocity $U_e(x)$ from a slender-body potential-flow solve of the fuselage geometry. The two-stage strategy uses Thwaites (1949) for the laminar leading segment and Head (1958) for the turbulent aft segment, with a Michel or $e^N$ transition criterion between them. Thwaites' method reduces closure to a single quadrature

$$\theta^{2}(x)\,U_e^{6}(x) \;=\; 0.45\,\nu\,\int_0^{x}\!U_e^{5}(\xi)\,d\xi, \tag{12}$$

with Curle–Skan polynomial closures for $H(\lambda)$ and $C_f(\lambda)$ where $\lambda = (\theta^{2}/\nu)\,dU_e/dx$; separation at $\lambda = -0.09$. Head's entrainment method carries two coupled ODEs $(\theta, H_1)$ with $H_1 = (\delta-\delta^{*})/\theta$, the entrainment law $d(U_e\theta H_1)/dx = U_e F(H_1)$, and the Ludwieg–Tillmann $C_f = 0.246\cdot 10^{-0.678 H}\,\text{Re}_\theta^{-0.268}$. Integrate with `scipy.integrate.solve_ivp(method='LSODA')` to handle stiffness near separation.

The kinetic-energy thickness $\theta^{*}$ is reconstructed from a Drela/Giles-style closure $H^{*}(H, \text{Re}_\theta)$ lifted from XFOIL/MSES, giving Drela-consistent $\theta^{*} = H^{*}\theta$ without implementing the full three-equation lag solver. Total dissipation follows the identity $\Phi = \int 2 C_{\mathcal D}\,U_e^{3}\,dA_{\text{wet}}$ over the fuselage, which equals $D_{\text{fus}}\,U_\infty$ to good approximation.

**Fuselage axisymmetry** is handled via the Mangler transformation, $\bar x = L^{-2}\int_0^x r^{2}(\xi)\,d\xi$, applied over the last ~20% of the fuselage where tail-cone contraction ($r/r_{\max}$ from 1.0 to ~0.3) accelerates $\theta$ growth by 10–30% relative to the 2-D flat-plate estimate. This correction materially affects the ingested momentum and must be included. **Wake extension** downstream of the trailing edge uses the Squire–Young closure $\theta_\infty = \theta_{TE}\,(U_{TE}/U_\infty)^{(H_{TE}+5)/2}$, sizing $\Phi_{\text{wake}}$ for the podded reference. Compressibility corrections at cruise $M_\infty \approx 0.78$ use the Van Driest II transformation on $C_f$.

### 3.2 The two-case comparator

The power-balance BLI benefit is computed by direct evaluation of both cases from a shared parameter set, avoiding the closed-form approximations that hide double-counting:

1. Compute $D_{\text{fuselage}}$, $\Phi_{\text{surf}}$, $\Phi_{\text{wake}}$ from IBL + Squire–Young.
2. **Podded case:** solve isentropic fan with undisturbed inlet ($p_{t1} = p_{t\infty}$); iterate mass flow so $\dot m(V_j - V_\infty) = F_{\text{required}}$; compute $P_K^{\text{pod}} = \dot m\,c_p T_{t\infty}\,[\text{FPR}^{(\gamma-1)/\gamma} - 1]$.
3. **BLI case:** mass-average the ingested-annulus inlet total pressure and enthalpy from a $(y/\delta)^{1/n}$ profile (default $n = 7$) at $s_{TE}$, capturing fraction $f_{BLI}$ of the BL dissipation. Solve the fan with the degraded inlet at the same FPR; iterate mass flow to hold net-force balance including the ingested stream tube.
4. $\text{PSC} = 1 - P_K^{\text{BLI}}/P_K^{\text{pod}}$; decompose via Drela's ledger into jet/surface/wake contributions à la Hall 2017.
5. Turboelectric net PSC follows (11).

Expected PSC: 4–12% for realistic aft-fuselage BLI, matching STARC-ABL and D8 numbers.

### 3.3 The recommended Python stack: minimal, reproducible, defensible

**Recommended minimal stack:** `numpy`, `scipy`, `matplotlib`, `SALib`, `chaospy`, `pandas`, `pyyaml`, `joblib`, `pytest`. Nothing else is needed. Justifications for the omissions matter as much as the inclusions:

`scipy.integrate.solve_ivp` (LSODA for stiffness) handles the Head IBL ODEs; `scipy.integrate.cumulative_trapezoid` handles the Thwaites quadrature and dissipation integrals; `scipy.optimize.brentq` handles the fan mass-flow root-find and the BLI operating-point solve. These cover 100% of the BLI/power-balance numerics with stable, well-documented APIs.

**pyCycle** (NASA, OpenMDAO-based) is actively maintained (v4.4.0 Oct 2025) but the maintainers themselves state that "the docs are nearly non-existent," an AI-generated DeepWiki is its *de facto* documentation, and it is engineered for full multi-spool cycle balancing — dramatically overkill when a single-stage fan with FPR as input is 20 lines of scipy. Skip it for the first paper. **OpenMDAO** delivers analytic derivatives valuable only for gradient-based MDO; for embarrassingly parallel Sobol/UQ sweeps, `joblib.Parallel` beats the cost-of-entry. Skip. **SUAVE** is stale (last commit Feb 2024, tutorials from 2022) and Breguet is a five-line integral. Skip. **AeroSandbox** is high-quality but has no power-balance module; keep it as an optional import for XFOIL-based $\theta,H$ sanity checks in validation. **JAX/autograd** offer nothing for \<10 design variables with cheap evals. Skip.

**SALib** (Herman & Usher, actively maintained) delivers Saltelli-sampled Sobol' indices; use $N = 2^{10} = 1024$ as a pilot base sample (~15k evaluations for 7 inputs) and $N = 2^{14} = 16384$ for the production run (~295k evaluations, ~4 hours single-threaded at ~50 ms per BLI evaluation, minutes with joblib). **chaospy** (v4.3.13, actively maintained) provides polynomial-chaos surrogates: for $D = 6\text{–}8$ inputs at order 3, ~100–500 collocation points via Smolyak sparse quadrature is 100× cheaper than Saltelli-Sobol and is the natural surrogate for the moment/percentile/Sobol'-from-PCE analysis. Cross-check top-3 total-order indices against SALib as a sanity gate.

### 3.4 Parametric atlas construction

Sweep design: primary heatmaps in $(f_{BLI}, \text{FPR})$ at fixed Mach/altitude/BL thickness, and in $(f_{BLI}, \delta/r_{TE})$; 1-D slices for diminishing-returns curves showing saturation near $f_{BLI} \approx 0.7\text{–}0.8$. Recommended grids: 41×41 for 2-D heatmaps (~1700 evals, seconds), 501 points for 1-D slices. Cache all outputs as Parquet for reproducibility.

Input distributions for UQ: Uniform over design ranges (`f_BLI`, `FPR`); Normal or Beta for uncertain physics parameters (`η_pol_fan ~ N(0.92, 0.02)`, `n_powerlaw ~ N(7, 1)`, distortion penalty `[0, 5%]`, electrical chain `η_elec ~ N(0.92, 0.02)`). PCE surrogate → analytical moments and percentiles → cross-validation by 10⁴-sample Monte-Carlo on the surrogate → reported as **PSC and Δblock-fuel PDFs with 5th/50th/95th percentile bands, with published point estimates (Uranga 8.2%, Yildirim 3–6%, Welstead 5–7%) overlaid.**

### 3.5 Verification and validation

**Verification** against analytical benchmarks: laminar flat plate Blasius (Thwaites gives $\theta = 0.671\sqrt{\nu x/U_\infty}$, target $C_f$ error \<1%); turbulent 1/7-power flat plate (Head, $C_f = 0.0592\,\text{Re}_x^{-1/5}$, target error \<3%); Falkner–Skan wedge for $m \in \{-0.09, 0, 0.5, 1.0\}$; method-of-manufactured-solutions on the ODE system for p-th-order convergence in $\Delta x$.

**Validation** with explicit comparison rules for each reference: reproduce Smith 1993 wake-ingestion analytic limit within 2%; reproduce Uranga D8 8.2% ± 0.8% at zero net streamwise force with equal-nozzle-area rule; bracket STARC-ABL between Welstead-Felder (2016 conceptual) and NASA/TM-20210016661 (2022 canonical) and Yildirim 2022 (isolated-propulsor RANS). **Do not force a single global match — publish a validation table showing each reference's rule and this-work's value with UQ bands.** Reviewers reward diagnostic honesty over spurious agreement.

---

## 4. Concrete execution plan

Total: ~22 calendar weeks at ~20 h/week ≈ 5–5.5 months. Effort is front-loaded on Phases 2–3 (the technical core) and Phase 7 (validation, where reviewers attack).

**Phase 1 — Literature consolidation and specification freeze (2 weeks, 30–40 h).** Annotated Zotero/BibTeX library organized by (a) power-balance/exergy formulations, (b) BLI benefit quantification, (c) configuration studies, (d) integral BL methods. `SPEC.md` freezing baseline: single-aisle STARC-ABL class, cruise M0.785 at FL350, $\text{Re}_L \approx 1.5\times 10^{8}$, fuselage $L = 37$ m, $D = 3.76$ m, mission ranges {500, 1500, 3000} nmi. Locked validation targets with citations. **Risk:** scope creep and validation-target ambiguity; mitigate by treating `SPEC.md` as immutable except by documented amendment.

**Phase 2 — Integral boundary-layer solver (3 weeks, 60–75 h).** `ibl/` package with Thwaites, Head, transition, Mangler, dissipation, wake modules. Verification notebook against Blasius, 1/7-power, Ludwieg–Tillmann APG, axisymmetric cylinder. pytest tolerances baked in. **Risk:** transition location strongly affects downstream $\theta$; parametrize $x_{tr}$ as a UQ variable rather than hiding it. **Make the repo public at Phase 2** so that JOSS's ≥ 6 months of public dev history is satisfied before submission.

**Phase 3 — Power-balance BLI benefit core (3 weeks, 60–80 h).** `powerbalance/` module: Drela 2009 ledger, `smith1993.py` closed-form, `hall2017.py` parametric decomposition, `comparator.py` two-case wrapper enforcing a shared `ControlVolume` object. Automated bookkeeping-residual test that fails CI if the three formulations disagree by more than a tolerance on Drela's §V toy actuator-disk example. **Risk:** sign conventions and CV inconsistencies (see pitfalls below).

**Phase 4 — Propulsion + fuel-burn chain (2 weeks, 40–50 h).** `propulsion/fan.py` (1-D isentropic fan with FPR-dependent $\eta_{\text{fan}}$ and parametric distortion penalty $\delta\eta_{\text{distort}}$); `propulsion/turboelectric.py` (generator/inverter/cable/motor chain, TSFC table); `mission/breguet.py`. Reproduce Welstead & Felder's 3500-nmi baseline within a few percent with BLI disabled. **Risk:** double-counting propulsive efficiency already inside $P_K$; mitigate with an explicit named-intermediate power-flow diagram enforced in code.

**Phase 5 — Parametric benefit atlas (2 weeks, 30–40 h).** `studies/atlas.py` sweeping $f_{BLI} \in [0,1]$, FPR $\in [1.2, 1.5]$, BL thickness (via $x_{tr}$ or $\text{Re}_\theta$), fuselage fineness, Mach $\in [0.72, 0.82]$, range. 2-D heatmaps, saturation curves, Pareto fronts. Cached Parquet/netCDF. **Risk:** non-physical results in separation corners; mark each grid cell with a validity flag from the Phase 2 solver and grey out invalid regions.

**Phase 6 — Sensitivity + UQ (2.5 weeks, 45–60 h).** SALib Saltelli Sobol' with $N = 1024$ (pilot) → $2^{14}$ (production); chaospy PCE surrogate at order 3 with Smolyak sparse quadrature; MC-UQ (LHS, $N = 10^{4}$) on the surrogate for PSC and Δfuel-burn PDFs, 5/50/95 percentiles. Figure: Sobol' $S_T$ bar chart ranked by dominant parameters; UQ histogram with published point estimates overlaid. **Risk:** if a single BL solve is >1 s, Saltelli becomes costly; vectorize IBL, add joblib caching, run Morris screening first to prune inputs.

**Phase 7 — Validation (2 weeks, 40–50 h).** Three notebooks: `smith1993.py` (Smith Fig. 4/5 curves), `uranga_d8.py` (8.2% ± 0.8% with equal-nozzle-area rule and the 5.2/2.4/0.6 decomposition), `starc_abl.py` (bracket Welstead-Felder and Yildirim). Publish the validation table with source/metric/rule/reference/this-work/notes columns.

**Phase 8 — Writeup, code release, JOSS companion (2.5 weeks, 50–65 h).** Paper draft ~9000 words (AIAA-style ~10–12k), figures at publication resolution, GitHub repo with BSD-3 license, `pyproject.toml` + `uv.lock`, Dockerfile, GitHub Actions CI on Python 3.11–3.12 across ubuntu/macos, Zenodo DOI on v0.1.0. JOSS companion `paper.md` (~700–1000 words) with Summary, Statement of Need, State of the Field, Software Design, Research Impact, AI-usage Disclosure per JOSS 2025 scope.

**Phase 9 — Submission (1 week + review).** arXiv preprint on submission day (physics.flu-dyn primary, cs.CE cross-list); *J. Propulsion & Power* via ScholarOne; cover letter, suggested reviewers from the citation graph, data-availability statement.

### 4.1 Technical pitfalls to watch, with mitigations

**Thrust/drag bookkeeping ambiguity in BLI.** The dividing streamline between "airframe wake" and "engine jet" is a definitional choice, not physical; different authors get different numbers for the same experiment. *Mitigation:* adopt the power balance from the outset. Report PSC exclusively at equal net streamwise force. In supplementary material, map power-balance PSC to conventional TSFC for legacy comparison.

**Consistent control-volume definitions.** Pitfalls: CV that clips the jet before mixing completes; different downstream extents between cases; lateral boundaries where $p \ne p_\infty$. *Mitigation:* one canonical CV extending to a Trefftz plane where $(u-U_\infty)/U_\infty < 10^{-3}$; encode the CV as a first-class object with an assertion that both cases share it.

**Double-counting of dissipation.** Charging the airframe with wake-dissipation drag in a "clean" polar, then also crediting BLI with wake recovery. *Mitigation:* within a case, never mix formulations. In the BLI case, compute total $P_K$ directly from the power balance — do not compute drag at all.

**Ingestion fraction definition.** Four common conventions (mass, area, dissipation, momentum-defect) give numerically different numbers for the same case. *Mitigation:* standardize on **dissipation fraction $f_\Phi$** (aligned with Hall 2017); publish cross-conversion table.

**BL model validity — flat plate vs. real fuselage.** Real fuselages have mild favorable gradients over the mid-section, strong adverse gradients on the tail cone with separation risk, and axisymmetric convergence. *Mitigation:* axisymmetric IBL with Mangler; $U_e(x)$ from slender-body potential-flow solve, not $U_e = U_\infty$; flag $H > 2.4$ as invalid; publish a validity envelope in (fineness × Mach × $f_{BLI}$) space; exclude wing-body junction and empennage effects and state so explicitly.

**Fan/BLI coupling ignored.** Real BLI fans lose 1–3 points of $\eta_{\text{fan}}$ to inlet distortion (Fidalgo/Hall 2012, Gray 2018). *Mitigation:* include $\delta\eta_{\text{distort}}(f_{BLI})$ as an explicit input with defensible default (1.5% at $f_{BLI} = 0.5$ rising to 3% at 1.0); sweep in UQ; cite as a known limitation where higher-fidelity coupling is required.

**Compressibility at cruise Mach.** *Mitigation:* Van Driest II on $C_f$; Cebeci-Smith/Squire-Young on $H, \theta$; validate against turbulent flat-plate correlations at $M = 0.7\text{–}0.9$.

**Validation traps — the STARC-ABL number is not a scalar.** Yildirim 2022's ~3% and Welstead-Felder's 7–12% measure different things at different FPRs with different distortion assumptions. *Mitigation:* publish per-rule validation, not a global forced match. Uranga's 8.2% is a specific wind-tunnel measurement of mechanical flow power at zero net force with equal nozzle area — reproduce that exact rule.

---

## 5. Paper framing and publishability

### 5.1 The four-claim novelty package

The paper is not a new theory. Its novelty is packaged as four ordered claims. **First**, the Smith 1993, Drela 2009, and Hall 2017 formulations are shown to be consistent limits of a single Python framework, with automated bookkeeping-residual checks — no prior open-source implementation unifies these. **Second**, it is the first open-source, fully reproducible implementation of the power-balance BLI benefit model; MDO Lab and MIT high-fidelity CFD chains are not open-source or reproducible outside their groups. **Third and most defensible**, it delivers formal global sensitivity indices and MC-UQ across the current published parameter ranges — a treatment absent from the peer-reviewed literature. **Fourth**, a single parametric benefit atlas lets a conceptual designer read off expected PSC with confidence bands for a chosen $(f_{BLI}, \text{FPR}, \text{mission})$ triple.

### 5.2 Target venue

Primary: **AIAA *Journal of Propulsion and Power***. This is where Smith 1993, Hall 2017, and Epstein 2019 live — the direct citation neighborhood, ~10–12k-word full paper, 4–8 month cycle. Second: **AIAA *Journal of Aircraft*** (Welstead 2016, Yildirim 2022, Blumenthal 2024). Third: **Aerospace Science and Technology** (Elsevier, ~5.0 IF, hybrid OA, broader scope). arXiv (physics.flu-dyn, cs.CE) preprint is mandatory on submission day. JOSS software companion runs in parallel after journal acceptance — start the public repo at Phase 2 for the ≥6-month history JOSS 2025 requires.

Working titles (AIAA ≤12-word limit): *"How Robust Is the BLI Benefit? Global Sensitivity and Uncertainty Quantification of a Low-Order Power-Balance Model"* is the strongest — it emphasizes the sensitivity/UQ novelty and asks a specific question the paper answers. Alternates: *"Quantifying the Boundary-Layer-Ingestion Propulsion Benefit under Uncertainty: A Reduced-Order Power-Balance Study"* and *"Reconciling Smith, Drela, and Hall: A Reproducible Low-Order BLI Benefit Assessment."*

### 5.3 The eight figures that make the paper

Fig. 1 is the schematic of the power-balance CV annotated with $\Phi_{\text{surf}}, \Phi_{\text{jet}}, \Phi_{\text{wake}}$ and the ingested BL stream tube. Fig. 2 is the IBL verification (Blasius, 1/7-power) with error bars. Fig. 3 is the dissipation-decomposition bar chart against Uranga's 5.2/2.4/0.6 D8 split. **Fig. 4 (headline) is the 2-D heatmap of PSC vs. $(f_{BLI}, \text{FPR})$** with the "practical envelope" contour overlaid. Fig. 5 is the diminishing-returns curve showing saturation near $f_{BLI} \approx 0.7\text{–}0.8$. Fig. 6 is the Sobol' $S_T$ bar chart — **expected finding: the distortion penalty and electrical-chain efficiency dominate over $f_{BLI}$ once $f_{BLI} > 0.5$, which reframes what BLI research should prioritize.** Fig. 7 is the UQ histogram with 5–95th percentile bands and Uranga/Yildirim/Welstead point estimates overlaid. Fig. 8 is fuel-burn Δ vs. mission range at three ingestion fractions. Table 1 is the validation table; Table 2 is the nomenclature/ingestion-fraction cross-conversion.

### 5.4 Section outline

Abstract (150–200 words, stating PSC range, dominant sensitivities, code DOI). §1 Introduction — decarbonization motivation, BLI history, verified gap, four contributions. §2 Method — 2.1 power balance (Drela), 2.2 Smith wake ingestion, 2.3 Hall decomposition, 2.4 unification and residual test, 2.5 IBL with compressibility and Mangler, 2.6 fan and turboelectric chain, 2.7 mission and fuel burn. §3 Verification & Validation. §4 Parametric Study (the atlas). §5 Sensitivity and UQ. §6 Discussion — comparison with CFD, limitations (fan-distortion coupling, tail-cone separation, 3-D effects), when the low-order tool suffices. §7 Conclusions. Appendix A: bookkeeping residual proof for the 2-D actuator disk. Appendix B: nomenclature and ingestion-fraction cross-conversion.

### 5.5 Reproducibility strategy

Repository layout: `bli-power-balance/{src/blipb, tests, docs, notebooks, studies, paper, data, .github/workflows}`. License: BSD-3 (OSI, JOSS-eligible, permissive). Env pinning: `pyproject.toml` + `uv.lock` with a `pixi.toml` scientific-stack alternative and a Dockerfile fallback. Testing: pytest with coverage ≥85%, CI across Python 3.11–3.12 on ubuntu/macos. Notebooks `00_quickstart` through `05_reproduce_paper_figures` — every paper figure regenerable by one cell. Cached sweep results as Parquet in `data/` (~50 MB, committable); anything larger to Zenodo. Zenodo DOI minted on v0.1.0 and cited in the paper. AI-usage disclosure per JOSS and Elsevier policy.

### 5.6 What makes this a tidy, publishable result

Five things: **honest scope** (state clearly this is low-order; the reviewer community respects that); **transparent limitations** (explicit section on fan-airframe pressure feedback, tail-cone separation dynamics, 3-D and unsteady effects, off-design); **clean validation** (do not fudge to match; if the model gives 6.5% where D8 measured 8.2%, publish and diagnose — likely axisymmetric wake modeling — reviewers reward diagnosis over spurious agreement); **novel insight** (the Sobol'+UQ output is the paper's defensible original contribution and its most quotable finding); **genuine openness** (full code, full data, full notebooks, reproducible from a fresh clone with `uv sync && pytest && jupyter nbconvert --execute`). This differentiates the paper permanently in a subfield dominated by proprietary CFD chains.

---

## Conclusion

The BLI subfield in 2025–2026 is dominated by high-fidelity CFD studies (Yildirim, Chau, Anibal, Sabnis) that are more physically accurate than any low-order model can be — and by conceptual studies that quote point estimates without uncertainty. **Neither camp has published rigorous global sensitivity analysis or uncertainty quantification of the BLI benefit itself; neither has released a reproducible open-source implementation of the Drela power-balance method.** These two omissions are the seams into which a single-researcher, Python-only, five-month project can insert a genuinely novel and defensible contribution. The physics is settled (Drela 2009; Hall/Uranga/Sabnis 2017–2023), the tooling is mature (scipy, SALib, chaospy), the validation targets are documented (Uranga D8 8.2% ± 0.8%; NASA/TM-20210016661 STARC-ABL 3.4%), and the venue exists (*J. Propulsion & Power* with a JOSS software companion). The paper's most quotable finding — likely that fan-distortion penalty and turboelectric-chain efficiency dominate BLI benefit variance once ingestion fraction exceeds ~0.5 — would reframe what BLI research should prioritize, and it is a finding only a systematic parametric+UQ study can produce. The proposal above lays out the physics, methodology, tooling, phased build plan, pitfall mitigations, and paper framing needed to execute it end-to-end.