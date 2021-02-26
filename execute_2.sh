#! /bin/bash

opt_tr="--tr even"
program="python3 exec_conductivity.py"

$program -i 0 -a 1 -b 0 $opt_tr
$program -i 1 -a 0 -b 0 $opt_tr
$program -i 2 -a 1 -b 1 $opt_tr
$program -i 2 -a 0 -b 0 $opt_tr
$program -i 1 -a 1 -b 1 $opt_tr
$program -i 0 -a 0 -b 1 $opt_tr

$program -i 0 -a 0 -b 0 $opt_tr
$program -i 1 -a 1 -b 0 $opt_tr
$program -i 2 -a 1 -b 0 $opt_tr
$program -i 2 -a 0 -b 1 $opt_tr
$program -i 1 -a 0 -b 1 $opt_tr
$program -i 0 -a 1 -b 1 $opt_tr


$program -a 0 -b 0 --mode c $opt_tr
$program -a 1 -b 1 --mode c $opt_tr

$program -a 1 -b 0 --mode c $opt_tr
$program -a 0 -b 1 --mode c $opt_tr


