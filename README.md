# 2D-AFM-Top
### This repository will contains some tight binding simulations of AFM and FM
### materials such as:

###         1.- CrAs2
###         2.- MnX3: Noncollinear AFM in the Kagome Lattice

### We will use the pythb library, cause its simplicity in working
### with topological quantities, like berry curvatures, and Chern numbers.

##  The code is structured as follows.
### 1.-sim_tb.py
###   Module containing the main Class of a Tight-Binding simulation.
###   This class needs a config file, and a second file containing the hoppings of the tight binding model.
###   The library pyTB is used for constructing and diagonalizing the Hamiltonian.

### 2.- bz_utilities
###   Module containing some utilities needed to perform calculations in the fisrt Brillouin Zone.

### 3.- plot_utils.py
###   Module with some utilities in the manipulations of plots (matploitlib)

### 4.- toy_models.py
###   This module contains useful functions for initialize the input files (config and hoppings),
###   for som toy models such as: 
###     (i) CrAs2 with s-electrons.
###     (ii) s-d model in noncollinear AFM kagome lattice.


### Moreover, there are some notebooks in which some specific (and less structured) tasks are performed.
###  5.- Testing_pytb.ypinb
###   Some testing involving the creation of the model, ploting of the bands, and calculations of berry phases in
###   the toy model of CrAs2.
###  6.- Kagome.ipynb
###   Some calculations in the noncollinear AFM kagome lattice.



