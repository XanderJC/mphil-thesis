# MPhil Thesis

Supplementary code for my [MPhil Machine Learning and Machine Intelligence](https://www.mlmi.eng.cam.ac.uk) thesis: 

## Interpretable Policy Learning


Last Updated: 20 August 2020

Code Author: Alex J. Chan (ajc340@cam.ac.uk)


Specifically this repo contains Jupyter notebook scripts to run both Variational DIPOLE and InterPoLe (algorithms 1 and 2) on a synthetic example problem (privacy regulations prevent the medical data used in the thesis being available here).

We also include separately a model class for our soft decision tree architecture that can be generally applied in supervised learning - we demonstrate it on the Iris data set

InterPoLe and the soft decision tree are written using the [Autograd](https://github.com/HIPS/autograd) auto-diff framework. This is a slightly outdated framework and runs relatively slowly but is quite nice to use. For Variational DIPOLE I transitioned to [JAX](https://github.com/google/jax), the successor of Autograd and consequently runs significantly faster, though it will take a moment to compile for XLA.


### Synthetic Data Examples