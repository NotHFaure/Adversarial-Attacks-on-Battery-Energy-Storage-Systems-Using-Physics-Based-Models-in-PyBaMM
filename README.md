# Adversarial Attacks on Battery Energy Storage Systems Using Physics-Based Models in PyBaMM

This repository contains the code, data, and literature for the research paper titled **"Adversarial Attacks on Battery Energy Storage Systems Using Physics-Based Models in PyBaMM"**, co-authored by **Alaa Selim, Harrison Faure, Huadong Mo, Hemanshu Pota, and Daoyi Dong**.

## Abstract

Battery Energy Storage Systems (BESS) are critical for modern energy infrastructure. However, they are vulnerable to adversarial attacks targeting key battery parameters such as current, voltage, and temperature. In this work, we simulate attacks using the Python Battery Mathematical Modelling framework (PyBaMM) and the Doyle-Fuller-Newman (DFN) model. The study focuses on how noise and perturbations affect terminal voltage, revealing potential vulnerabilities and discussing defensive strategies. This research aims to contribute to the security and reliability of BESS under adversarial conditions.

## Repository Structure

- **Code**: Python implementations using the PyBaMM framework.
- **Data**: Large datasets required for simulations. Not included in this repository — see "Links to External Files" below.
- **Literature**: References and related works for adversarial attacks and battery modeling. The co-authored paper (`Pybamm_Journal_Paper (3).pdf`) and supporting planning and report documents are included in this repository under `Reports/`.

## Links to External Files

Due to file size constraints, some resources (datasets, literature, implementation code) are hosted externally rather than committed to this repository. Data is available on request.

## Key Components

1. **DFN Model Implementation**
   - The **Doyle-Fuller-Newman (DFN)** model simulates lithium-ion battery behavior under adversarial attacks.
   - Parameters such as current, temperature, and SEI (Solid Electrolyte Interphase) resistance are manipulated to observe impacts on State of Charge (SoC), State of Health (SoH), and terminal voltage.

2. **Adversarial Attack Simulation**
   - Both **single-window** and **multi-window** attack scenarios are considered.
   - Perturbations are applied to current, temperature, and SEI resistance with varying noise levels and spike probabilities.

3. **Sensitivity Analysis**
   - A detailed sensitivity analysis explores how perturbations affect battery performance, highlighting vulnerabilities and potential attack vectors.

For the latest updates, please check the **sensitivity analysis** and **adversarial attack files**.

## Acknowledgements

Listed in the order given in the paper:

- Alaa Selim
- Harrison Faure
- Huadong Mo
- Hemanshu Pota
- Daoyi Dong

## Repository status

This repository holds research code and materials supporting a co-authored paper. The research work dates from 2024 and is not maintained.

In 2026 the repository underwent a separate documentation and security-hygiene pass, which did not alter any research code, data or result. That pass removed externally-hosted data links and documents carrying a student identifier from the working tree, and added `.gitattributes` and `.gitignore`. It did not rewrite history, so content removed at that time remains retrievable from commits prior to `26ac331`.

## Known limitations

- Results in this repository are not summarised numerically; no accuracy, error or performance figures are recorded here.
- The externally linked data referenced above is not durably hosted.
- Content removed from the working tree in 2026 (externally-hosted data links and documents carrying a student identifier) remains retrievable from git history prior to commit `26ac331` — it was not deleted from history.
