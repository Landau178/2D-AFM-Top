#! /bin/bash

opt_tr="--tr even"
opt_soc="--t2 0.0"

program="python3 exec_conductivity.py $opt_tr $opt_soc"





$program -i 0 -a 1 -b
$program -i 1 -a 0 -b
$program -i 2 -a 1 -b
$program -i 2 -a 0 -b
$program -i 1 -a 1 -b
$program -i 0 -a 0 -b

$program -i 0 -a 0 -b
$program -i 1 -a 1 -b
$program -i 2 -a 1 -b
$program -i 2 -a 0 -b
$program -i 1 -a 0 -b
$program -i 0 -a 1 -b


$program -a 0 -b 0 --mode
$program -a 1 -b 1 --mode

$program -a 1 -b 0 --mode
$program -a 0 -b 1 --mode


