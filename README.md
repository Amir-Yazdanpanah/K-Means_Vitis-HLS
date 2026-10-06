# K-Means Hardware Acceleration with Vitis HLS

This project implements the K-Means clustering algorithm in a form suitable for High-Level Synthesis (HLS) and explores performance-oriented FPGA optimizations. It includes a baseline synthesizable implementation and two optimized variants designed to improve throughput and resource utilization in Vitis HLS.

The repository is structured as an academic / research-oriented implementation for hardware acceleration, combining:

- a synthesizable C/C++ K-Means kernel
- Vitis HLS project scripts
- input datasets for small 2D clustering examples
- synthesis reports for benchmarking and comparison

## Overview

K-Means partitions a set of points into K clusters by repeatedly:

1. assigning each point to its nearest centroid,
2. recomputing centroid positions using the mean of assigned points,
3. checking convergence by centroid movement or assignment stability.

In this implementation, the algorithm is ported to HLS-friendly C++ and optimized for FPGA-style dataflow execution, using techniques such as:

- local buffering of data and centroid arrays,
- loop pipelining,
- unrolling of innermost computations,
- memory-interface tuning (`m_axi` and `axis`),
- array partitioning for parallel access.

## Repository Structure

```text
.
├── Base/
│   ├── KMEANS.h
│   ├── KMEANS_HLS_base.cpp
│   ├── test_KMEANS_base.cpp
│   ├── input2D_2clusters.in
│   ├── run_KMEANS_HLS_base.tcl
│   └── csynth.rpt
├── Optimized_m_axi/
│   ├── KMEANS.h
│   ├── KMEANS_HLS_base.cpp
│   ├── test_KMEANS_base.cpp
│   ├── input2D_2clusters.in
│   ├── run_KMEANS_HLS_base.tcl
│   └── csynth.rpt
├── Optimized_axis/
│   ├── KMEANS.h
│   ├── KMEANS_HLS_base.cpp
│   ├── test_KMEANS_base.cpp
│   ├── input2D_2clusters.in
│   ├── run_KMEANS_HLS_base.tcl
│   └── csynth.rpt
├── README.html
└── README.md
```

### Folder descriptions

- `Base/`
  - Baseline implementation.
  - Converts the algorithm into a synthesizable kernel using standard HLS interfaces.
  - Good reference point for understanding the original functional flow.

- `Optimized_m_axi/`
  - Adds memory-oriented optimization strategies.
  - Uses `m_axi` interfaces and local buffers to improve memory throughput.
  - Includes explicit pipelining and array partitioning.

- `Optimized_axis/`
  - More advanced implementation using streaming interfaces.
  - Uses `hls::stream` to move data efficiently through the kernel.
  - Demonstrates a more hardware-oriented architecture for FPGA acceleration.

## Algorithm Behavior

The top-level kernel function is `do_compute(...)`. It performs the following tasks:

1. Reads the dataset and centroid state.
2. Computes squared Euclidean distances for each point against all centroids.
3. Assigns each point to the nearest centroid.
4. Rebuilds cluster sums and re-estimates centroid locations.
5. Repeats until the maximum iteration count is reached or the centroid movement is below tolerance.

The kernel is implemented so that the classification and centroid update phases are suitable for HLS scheduling and hardware optimization.

## Input Format

The input files are plain text datasets with the following format:

```text
<number_of_points> <number_of_dimensions>
<x1> <y1>
<x2> <y2>
...
```

Example:

```text
20 2
376 178
-302 -107
522 601
-399 -347
...
```

This repository uses small synthetic 2D datasets for functional testing and synthesis evaluation.

## Execution and Synthesis

The project is intended to be used with Xilinx Vitis HLS.

### Run the HLS project

From any version folder (for example `Base/`):

```bash
vitis_hls -f run_KMEANS_HLS_base.tcl
```

This script does the following:

- creates a Vitis HLS project,
- sets the top function,
- adds source files and testbench files,
- sets the FPGA target,
- runs C simulation,
- performs synthesis (`csynth_design`),
- optionally enables co-simulation.

### Example project script

The Tcl files are structured like this:

```tcl
open_project -reset KMEANS_HLS_base
set_top do_compute

add_files KMEANS.h
add_files KMEANS_HLS_base.cpp
add_files -tb test_KMEANS_base.cpp
add_files -tb [glob ./*.in]

open_solution -reset "solution_KMEANS_HLS_base"
set_part {virtexuplusHBM}
create_clock -period 300MHz

csim_design -argv "input2D_2clusters.in 2 10 0 0 ./2D_2clusters.out"
csynth_design
```

The Tcl scripts include comments indicating where optimization flags and co-simulation can be enabled or disabled.

## Optimization Notes

The different implementations illustrate common HLS optimization strategies:

### 1. Baseline

- functional synthesizable version,
- good for correctness and reference behavior,
- simple memory behavior and moderate resource usage.

### 2. `m_axi` optimization

- adds explicit memory interfaces to improve external memory access,
- buffers data/centroids locally,
- uses loop pipelining and array partitioning to reduce latency and improve parallelism.

### 3. `axis` optimization

- uses streaming interfaces instead of direct RAM-centric transfers,
- improves throughput for streaming-like data patterns,
- more hardware-friendly for acceleration pipelines.

## Notes on the Designed Kernel

The implementation intentionally avoids expensive square-root calculations during centroid comparisons and instead compares squared distances:

```cpp
dist += (point - centroid)^2
```

This reduces computational cost while preserving nearest-neighbor behavior for cluster assignment. The final convergence test uses the squared distance magnitude for the centroid update check, and is then converted to a scalar distance when needed.

## Expected Use Cases

This project is suitable for:

- academic exploration of FPGA acceleration,
- teaching HLS design workflows,
- benchmarking memory and loop-level optimization techniques,
- understanding how a clustering algorithm maps to hardware.

## Limitations

This repository is intentionally focused on small 2D example datasets and HLS design experimentation. It is not a full production-grade ML pipeline and does not include:

- generalized input validation beyond basic project examples,
- large-scale dataset handling,
- distributed training or streaming cloud deployment,
- high-level Python or ML-framework integration.

## Suggested Next Steps

Possible extensions include:

- parameterizing the number of points and dimensions,
- supporting arbitrary K and larger datasets,
- benchmarking latency and resource usage against different FPGA targets,
- integrating a host-side C++ or Python driver for automated validation,
- improving readability and modularity for publication-quality code.

## License

This project is provided as an educational implementation for hardware acceleration and academic use. Please check your local institutional or course requirements before redistributing or publishing it.

## Credits

This repository reflects a course-style Vitis HLS assignment centered around K-Means optimization and FPGA acceleration. The structure and intent are similar to an educational hardware implementation project for clustering workloads.
