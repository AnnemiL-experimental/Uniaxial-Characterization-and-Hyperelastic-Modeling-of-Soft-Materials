# Materials Testing of Silicone Elastomer and Porcine Skeletal Muscle

_A study integrating uniaxial material testing with hyperelastic constitutive modeling._

This repository serves as a technical writing sample and research artifact, containing the complete analysis pipeline for a biomechanics laboratory study that characterizes the large-deformation mechanical behavior of a silicone elastomer and porcine skeletal Muscle under uniaxial tension and compression. The goal is to demonstrate a complete scientific workflow spanning experiment, data processing, constitutive modeling, and technical writing.

# Highlights

✔ Large-deformation mechanics (tension + compression)

✔ Engineering stress computation (First Piola–Kirchhoff)

✔ Viscoelastic relaxation analysis (compression hold)

✔ Ogden hyperelastic model fitting (N = 1 and N = 4)

✔ Comparison of synthetic vs. biological soft tissues

✔ Full scientific report included for reference

# Pipeline Overview

Raw Data → Preprocessing → Stress/Strain Computation → Model Fitting → Visualization → Report

# Repository Structure

├── README.md

├── report.pdf  

├── data/     

├── theory/ 

└── analysis/            

# Key Computations

Engineering (1st Piola–Kirchhoff) Stress:

  P = F / A₀

Stretch Ratio:

  λ = 1 + ΔL / L₀

Ogden Hyperelastic Model:

  W = Σ (μᵢ / αᵢ)(λ₁^{αᵢ} + λ₂^{αᵢ} + λ₃^{αᵢ} − 3)

# Fit modes

| Material | Mode        | Model | Terms |
| -------- | ----------- | ----- | ----- |
| Silicone | Tension     | Ogden | N = 4 |
| Silicone | Compression | Ogden | N = 1 |
| Muscle   | Compression | Ogden | N = 1 |

# Results Summary

Silicone exhibits isotropic hyperelasticity with strain stiffening.

Porcine muscle is significantly softer, heterogeneous, and viscoelastic.

Relaxation decay: Silicone ~4.5% vs. Muscle ~58.7%.

A four-term Ogden model fits tensile silicone with R² ≈ 0.997.

# Theoretical Modeling Supplement

To complement the experimental characterization, a separate theoretical analysis of the Mooney–Rivlin hyperelastic model was performed. The model was derived from its strain energy formulation, converted to the First Piola–Kirchhoff (engineering) stress for uniaxial tension and simple shear, and evaluated across parameter variations to illustrate nonlinear mechanical responses.

The modeling component strengthens the interpretation of soft material behavior and connects laboratory measurements with continuum mechanics.

Artifact available in:

theory/Mooney_Rivlin_Model.pdf

# Environment

Python 3.13.2

NumPy / SciPy / Matplotlib / Pandas


# Acknowledgements

Portions of the data processing and plotting code were developed with assistance from Gemini 3 Pro and ChatGPT. ChatGPT was also used during the preparation of the written report to clarify language and refine organization. The author conducted all analyses, made all modeling decisions, and provided all scientific interpretations.
