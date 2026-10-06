#include "KMEANS.h"
#include <iostream>
#include <cmath>

/*
Function readInput: gets number of points, dims, and the coordinate of each
point from file

Do not modify this function
*/
void readInput(char *filename, int *npoints, int *ndims, float **data) {
    // this part is slow. we can generate hidden test cases during the runtime to
    // avoid storage system io
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        fprintf(stderr, "File open error: %s.\n", filename);
        exit(-2);
    }

    if (fscanf(fp, "%d%d", npoints, ndims) < 2) {
        fprintf(stderr, "Error reading file: %s.\n", filename);
        exit(-2);
    }
    long n = *npoints * *ndims;
    *data = (float *)calloc(n, sizeof(float));
    if (*data == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(-1);
    }

    float *ptr = *data;

    for (long i = 0; i < n; i++) {
        if (fscanf(fp, "%f", ptr) <= 0) {
            fprintf(stderr, "Error reading file: %s.\n", filename);
            exit(-2);
        }
        ptr++;
    }
}

/*
Function writeResult: It writes in the output file the cluster of each sample
(point).

Do not modify this function
*/
void writeResult(int *classMap, int npoints, const char *filename) {
    FILE *fp = fopen(filename, "wt");
    if (fp != NULL) {
        for (int i = 0; i < npoints; i++) {
            fprintf(fp, "%d\n", classMap[i]);
        }
        fclose(fp);
    } else {
        fprintf(stderr, "Error writing file: %s.\n", filename);  // No file found
        fflush(stderr);
        exit(-3);
    }
}

/*

Function initCentroids: This function copies the values of the initial
centroids, using their position in the input data structure as a reference map.

Do not modify this function
*/
void initCentroids(const float *data, float *centroids, int *centroidPos, int ndims, int K) {
    for (int i = 0; i < K; i++) {
        int idx = centroidPos[i];
        memcpy(&centroids[i * ndims], &data[idx * ndims], (ndims * sizeof(float)));
    }
}

int main(int argc, char **argv) {
    /*
     * PARAMETERS
     *
     * argv[1]: Input data file
     * argv[2]: Number of clusters
     * argv[3]: Maximum number of iterations of the method. Algorithm termination
     * condition.
     * argv[4]: Minimum percentage of class changes. Algorithm
     * termination condition. If between one iteration and the next, the
     * percentage of class changes is less than this percentage, the algorithm
     * stops.
     * argv[5]: Precision in the centroid distance after the update. It is
     * an algorithm termination condition. If between one iteration of the
     * algorithm and the next, the maximum distance between centroids is less
     * than this precision, the algorithm stops.
     * argv[6]: Output file. Class
     * assigned to each point of the input file.
     * */
    if (argc != 7) {
        fprintf(stderr, "EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
        fprintf(stderr,
                "./KMEANS <Input Filename> <Number of clusters> <Number of "
                "iterations> <Number of changes> <Threshold> <Output data file>\n");
        fflush(stderr);
        exit(-1);
    }

    // use monotonic clock for timing
    struct timespec start, end;
    double duration;

    // ----------------------- Init Start ---------------------------
    clock_gettime(CLOCK_MONOTONIC, &start);
    // Reading the input data
    int npoints = 0, ndims = 0;
    float *data;
    readInput(argv[1], &npoints, &ndims, &data);

    // Parameters
    int K = atoi(argv[2]);
    int maxIterations = atoi(argv[3]);
    int minChanges = (int)(npoints * atof(argv[4]) / 100.0);
    float maxThreshold = atof(argv[5]);

    // mem allocs
    int *centroidPos = (int *)calloc(K, sizeof(int));
    float *centroids = (float *)calloc(K * ndims, sizeof(float));
    int *classMap = (int *)calloc(npoints, sizeof(int));

    if (centroidPos == NULL || centroids == NULL || classMap == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(-4);
    }

    // Init centroids
    srand(42);

    int *centroidSelected = classMap;  // use the memory temporarily
    for (int i = 0; i < K; i++) {
        for (;;) {
            int pos = rand() % npoints;
            if (centroidSelected[pos] == 0) {
                centroidSelected[pos] = 1;
                centroidPos[i] = pos;
                break;
            }
        }
    }
    for (int i = 0; i < K; i++) {
        centroidSelected[centroidPos[i]] = 0;
    }

    // Loading the array of initial centroids with the data from the array data
    // The centroids are points stored in the data array.
    initCentroids(data, centroids, centroidPos, ndims, K);

    printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], npoints, ndims);
    printf("\tNumber of clusters: %d\n", K);
    printf("\tMaximum number of iterations: %d\n", maxIterations);
    printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), npoints);
    printf("\tMaximum centroid precision: %f\n", maxThreshold);

    clock_gettime(CLOCK_MONOTONIC, &end);
    duration = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;
    printf("\nMemory allocation: %f seconds\n", duration);
    fflush(stdout);
    // ----------------------- Init End ---------------------------

    // ----------------------- Computation Start ---------------------------
    clock_gettime(CLOCK_MONOTONIC, &start);

    // set input parameters
    struct parameters p = {
                           .data = data
                        };
    struct results r = {.classMap = classMap, .centroids = centroids};

    // this is the entry point of your code.
    do_compute(data, classMap, centroids);

    for (int i = 0; i < 2; i++) {
        std::cout << "Cen [" << i << "] = " << centroids[i*2] << ", " << centroids[(i*2) + 1]<< std::endl;
    }
    for (int i = 0; i < 20; i++) {
        std::cout << "Classmap [" << i << "] = " << classMap[i] << std::endl;
        std::cout << "data [" << i << "] = " << data[i*2] << ", " << data[(i*2) + 1] << std::endl;
    }
    // timing of the do_compute function, including extra memory allocation.
    // You can measure and report the exact computation time only in your code,
    // but we will also check the time with other operations
    clock_gettime(CLOCK_MONOTONIC, &end);
    duration = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;
    printf("\nComputation: %f seconds", duration);
    fflush(stdout);
    //**************************************************
    // ----------------------- Computation End ---------------------------
    
    writeResult(classMap, npoints, argv[6]);

    // --------------- Memory deallocation Start ---------------------
    clock_gettime(CLOCK_MONOTONIC, &start);
    // Free memory
    free(data);
    free(classMap);
    free(centroidPos);
    free(centroids);
    // free(outputMsg);

    clock_gettime(CLOCK_MONOTONIC, &end);
    duration = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;
    printf("\n\nMemory deallocation: %f seconds\n", duration);
    fflush(stdout);
    // ---------------- Memory deallocation End -------------------
    return 0;
}
