#! /bin/bash

# python3 exec_SHC.py -i 0 -a 1 -b 0
# python3 exec_SHC.py -i 1 -a 0 -b 0
# python3 exec_SHC.py -i 2 -a 1 -b 1
# python3 exec_SHC.py -i 2 -a 0 -b 0
# python3 exec_SHC.py -i 1 -a 1 -b 1
# python3 exec_SHC.py -i 0 -a 0 -b 1

python3 exec_SHC.py -i 0 -a 0 -b 0
python3 exec_SHC.py -i 1 -a 1 -b 0
python3 exec_SHC.py -i 2 -a 1 -b 0
python3 exec_SHC.py -i 2 -a 0 -b 1
python3 exec_SHC.py -i 1 -a 0 -b 1
python3 exec_SHC.py -i 0 -a 1 -b 1


# python3 exec_SHC.py -a 0 -b 0 --mode c
# python3 exec_SHC.py -a 1 -b 1 --mode c

python3 exec_SHC.py -a 1 -b 0 --mode c
python3 exec_SHC.py -a 0 -b 1 --mode c



