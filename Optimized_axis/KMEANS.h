#pragma once

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cmath>
#include <hls_stream.h>  // Added for AXI Stream support

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#define ndims 2         // Dimensions per point
#define npoints 20      // Number of points
#define K 2             // Number of clusters
#define nclusters 2     // Number of clusters

// Function prototype with AXI Stream interfaces
void do_compute(
    hls::stream<float>& data_stream,   // AXI Stream for input data (floats)
    hls::stream<int>& class_stream,    // AXI Stream for output classes (ints)
    float* centroids                   // Centroids in DDR (via m_axi)
);