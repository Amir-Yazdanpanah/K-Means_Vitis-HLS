#pragma once

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cmath>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#define ndimss 2         // or pass this as a template parameter
#define npointss 20       // Number of points
#define KK 2             // Numeber of clusters
#define nclusters 2     // Numeber of clusters

// Hint: you might need to change some of these struct-s
struct parameters {
    float *data;
};

struct results {
    int *classMap;
    float *centroids;
};

void do_compute(
    float data[npointss * ndimss], 
    int classMap[npointss], 
    float centroids[KK * ndimss]);
