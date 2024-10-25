# Adversarial Attacks on Battery Energy Storage Systems Using Physics-Based Models in PyBaMM

This repository contains the code, data, and literature for the research paper titled **"Adversarial Attacks on Battery Energy Storage Systems Using Physics-Based Models in PyBaMM"**, co-authored by **Alaa Selim, Harrison Faure, Huadong Mo, Hemanshu Pota, and Daoyi Dong**.

## Abstract

Battery Energy Storage Systems (BESS) are critical for modern energy infrastructure. However, they are vulnerable to adversarial attacks targeting key battery parameters such as current, voltage, and temperature. In this work, we simulate attacks using the Python Battery Mathematical Modelling framework (PyBaMM) and the Doyle-Fuller-Newman (DFN) model. The study focuses on how noise and perturbations affect terminal voltage, revealing potential vulnerabilities and discussing defensive strategies. This research aims to contribute to the security and reliability of BESS under adversarial conditions.

## Repository Structure

- **Code**: Python implementations using the PyBaMM framework.
- **Data**: Large datasets required for simulations. *(Too large to commit, see links below)*
- **Literature**: References and related works for adversarial attacks and battery modeling. *(Too large to commit, see links below)*

## Links to External Files

Due to file size constraints, some resources are hosted externally. You can access them via the following links:

- [**Data**](https://1drv.ms/f/s!AqGwCfpEgvD9kuIDu4eBeZ7ZpdvlyA?e=aSjjHc) — Datasets for battery simulations.
- [**Literature**](https://1drv.ms/f/s!AqGwCfpEgvD9kuBToGrSkxt-mc3isw?e=eXxutK) — Relevant papers and literature on battery modeling and adversarial attacks.
- [**Implementations**](https://1drv.ms/f/s!AqGwCfpEgvD9hOQ-z-_Towqcs1zwcw?e=LsuxlR) — Source code for implementing the adversarial attacks.

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
