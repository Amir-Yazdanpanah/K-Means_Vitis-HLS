# Create a new Vitis HLS project
open_project -reset KMEANS_HLS_base
set_top do_compute

add_files KMEANS.h
add_files KMEANS_HLS_base.cpp

add_files -tb test_KMEANS_base.cpp

# Add data files (for simulation)
add_files -tb [glob ./*.in]

# Create a solution
open_solution -reset "solution_KMEANS_HLS_base"

# Set the target FPGA part (modify as needed)
set_part {virtexuplusHBM}

# Set clock
create_clock -period 200MHz

# Run C simulation
csim_design -argv "input2D_2clusters.in 2 10 0 0 ./2D_2clusters.out"

# Disable automatic pipelining (what does it change if you remove/comment this command?)
config_compile -pipeline_loops 0

# Run High-Level Synthesis (HLS)
csynth_design

# Run co-simulation (Attention, this might require a long time)
# cosim_design -argv "input2D_2clusters.in 2 10 0 0 ./2D_2clusters.out"


exit
