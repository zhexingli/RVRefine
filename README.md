# RVRefine
Numerical simulation and analysis code for refining exoplanet orbital period uncertainties using RV follow ups.

Prerequisite packages:
1. RadVel  (for radial velocity modeling)
2. Sklearn  (for data analysis using machine learning models)
3. PyGAM   (for data analysis using the Generalized Additive Model)

Files and workflow:
1. UncertaintySim.py
   Main simulation file. Executes simulations based on the input provided.
2. Collect.py
   Collect, retrieve, and store ALL the simulation data into one organized file for analysis.
3. ML_model.py
   File for analyzing the simulation data using the Random Forest model.
4. GAMFit.py
   File for analyzing the uncertainty evolution with respect to one simulation variable using the Generalized Additive Model.

