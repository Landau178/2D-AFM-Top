#! /bin/bash

opt_tr = "--tr even"

python3 exec_SHC.py -i 0 -a 1 -b 0 $opt_tr
python3 exec_SHC.py -i 1 -a 0 -b 0 $opt_tr
python3 exec_SHC.py -i 2 -a 1 -b 1 $opt_tr
python3 exec_SHC.py -i 2 -a 0 -b 0 $opt_tr
python3 exec_SHC.py -i 1 -a 1 -b 1 $opt_tr
python3 exec_SHC.py -i 0 -a 0 -b 1 $opt_tr

python3 exec_SHC.py -i 0 -a 0 -b 0 $opt_tr
python3 exec_SHC.py -i 1 -a 1 -b 0 $opt_tr
python3 exec_SHC.py -i 2 -a 1 -b 0 $opt_tr
python3 exec_SHC.py -i 2 -a 0 -b 1 $opt_tr
python3 exec_SHC.py -i 1 -a 0 -b 1 $opt_tr
python3 exec_SHC.py -i 0 -a 1 -b 1 $opt_tr


python3 exec_SHC.py -a 0 -b 0 --mode c $opt_tr
python3 exec_SHC.py -a 1 -b 1 --mode c $opt_tr

python3 exec_SHC.py -a 1 -b 0 --mode c $opt_tr
python3 exec_SHC.py -a 0 -b 1 --mode c $opt_tr



