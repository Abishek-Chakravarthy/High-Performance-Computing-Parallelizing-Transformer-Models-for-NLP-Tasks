#!/bin/bash

# Compile with gprof
g++ -pg -o hpc_profiler hpc.cpp -O2
./hpc_profiler
gprof hpc_profiler gmon.out > gprof_report2.txt
echo "gprof report generated."

# Compile with gcov
g++ -fprofile-arcs -ftest-coverage -O0 -g -o hpc hpc.cpp
./hpc
gcov -b -c hpc.cpp
echo "gcov report generated."

echo "All profiling completed."
