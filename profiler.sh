#! /bin/bash

opt_tr="--tr even"

program_even1="exec_conductivity.py --tr even --nk 20 -i 2 -a 0 -b 1"
program_even2="exec_conductivity.py --tr even --nk 50 -i 2 -a 0 -b 1"
program_even3="exec_conductivity.py --tr even --nk 100 -i 2 -a 0 -b 1"

program_odd1="exec_conductivity.py --tr odd --nk 20 -i 2 -a 0 -b 1"
program_odd2="exec_conductivity.py --tr odd --nk 50 -i 2 -a 0 -b 1"
program_odd3="exec_conductivity.py --tr odd --nk 100 -i 2 -a 0 -b 1"

kernprof -l -o even1.lprof $program_even1
kernprof -l -o even2.lprof $program_even2
kernprof -l -o even3.lprof $program_even3

kernprof -l -o odd1.lprof $program_odd1
kernprof -l -o odd2.lprof $program_odd2
kernprof -l -o odd3.lprof $program_odd3

rm line_profiler/*.txt

python3 -m line_profiler even1.lprof >>line_profiler/even1.txt
python3 -m line_profiler even2.lprof >>line_profiler/even2.txt
python3 -m line_profiler even3.lprof >>line_profiler/even3.txt

python3 -m line_profiler odd1.lprof >>line_profiler/odd1.txt
python3 -m line_profiler odd2.lprof >>line_profiler/odd2.txt
python3 -m line_profiler odd3.lprof >>line_profiler/odd3.txt

rm *.lprof
