#! /bin/bash

opt_soc="--t2 $1"
opt_tr="--tr even"


program="python3 exec_conductivity.py $opt_tr $opt_soc"


# Calcullate all components of spin conductivity
$program -i 0 -a 1 -b 0
$program -i 1 -a 0 -b 0
$program -i 2 -a 1 -b 1
$program -i 2 -a 0 -b 0
$program -i 1 -a 1 -b 1
$program -i 0 -a 0 -b 1

$program -i 0 -a 0 -b 0
$program -i 1 -a 1 -b 0
$program -i 2 -a 1 -b 0
$program -i 2 -a 0 -b 1
$program -i 1 -a 0 -b 1
$program -i 0 -a 1 -b 1

#Calculate all components of charge conductivity
$program -a 0 -b 0 --mode c
$program -a 1 -b 1 --mode c
$program -a 1 -b 0 --mode c
$program -a 0 -b 1 --mode c

