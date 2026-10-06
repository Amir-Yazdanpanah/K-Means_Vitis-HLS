#include "KMEANS.h"
#include <iostream>
#include <cmath>
#include <hls_stream.h>  // For AXI Stream support

// Undefine conflicting macros from KMEANS.h
#undef ndims
#undef npoints
#undef K

/*
Function readInput: gets number of points, dims, and the coordinate of each
point from file
*/
void readInput(char *filename, int *num_points, int *num_dims, float **data) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        fprintf(stderr, "File open error: %s.\n", filename);
        exit(-2);
    }

    if (fscanf(fp, "%d%d", num_points, num_dims) < 2) {
        fprintf(stderr, "Error reading file: %s.\n", filename);
        exit(-2);
    }
    long n = *num_points * *num_dims;
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
    fclose(fp);
}

/*
Function writeResult: It writes in the output file the cluster of each sample
*/
void writeResult(int *classMap, int num_points, const char *filename) {
    FILE *fp = fopen(filename, "wt");
    if (fp != NULL) {
        for (int i = 0; i < num_points; i++) {
            fprintf(fp, "%d\n", classMap[i]);
        }
        fclose(fp);
    } else {
        fprintf(stderr, "Error writing file: %s.\n", filename);
        fflush(stderr);
        exit(-3);
    }
}

/*
Function initCentroids: Copies values of initial centroids
*/
void initCentroids(const float *data, float *centroids, int *centroidPos, int dims, int clusters) {
    for (int i = 0; i < clusters; i++) {
        int idx = centroidPos[i];
        for (int d = 0; d < dims; d++) {
            centroids[i * dims + d] = data[idx * dims + d];
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
        fprintf(stderr,
                "./KMEANS <Input Filename> <Number of clusters> <Number of "
                "iterations> <Number of changes> <Threshold> <Output data file>\n");
        fflush(stderr);
        exit(-1);
    }

    // Timing variables
    struct timespec start, end;
    double duration;

    // ----------------------- Init Start ---------------------------
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Reading the input data
    int num_points = 0, dims = 0;
    float *data;
    readInput(argv[1], &num_points, &dims, &data);

    // Parameters
    int clusters = atoi(argv[2]);
    int maxIterations = atoi(argv[3]);
    int minChanges = (int)(num_points * atof(argv[4]) / 100.0);
    float maxThreshold = atof(argv[5]);

    // mem allocs
    int *centroidPos = (int *)calloc(clusters, sizeof(int));
    float *centroids = (float *)calloc(clusters * dims, sizeof(float));
    int *classMap = (int *)calloc(num_points, sizeof(int));

    if (centroidPos == NULL || centroids == NULL || classMap == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(-4);
    }

    // Init centroids
    srand(42);
    int *centroidSelected = classMap;  // use temporarily
    for (int i = 0; i < clusters; i++) {
        for (;;) {
            int pos = rand() % num_points;
            if (centroidSelected[pos] == 0) {
                centroidSelected[pos] = 1;
                centroidPos[i] = pos;
                break;
            }
        }
    }
    for (int i = 0; i < clusters; i++) {
        centroidSelected[centroidPos[i]] = 0;
    }

    initCentroids(data, centroids, centroidPos, dims, clusters);

    printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], num_points, dims);
    printf("\tNumber of clusters: %d\n", clusters);
    printf("\tMaximum number of iterations: %d\n", maxIterations);
    printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), num_points);
    printf("\tMaximum centroid precision: %f\n", maxThreshold);

    // Create AXI Stream interfaces
    hls::stream<float> data_stream;
    hls::stream<int> class_stream;

    // Write data to input stream (point-by-point, dimension-by-dimension)
    for (int i = 0; i < num_points; i++) {
        for (int d = 0; d < dims; d++) {
            data_stream.write(data[i * dims + d]);
        }
    }

    // ----------------------- Computation Start ---------------------------
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Call the accelerator with streams
    do_compute(data_stream, class_stream, centroids);

    // Read results from output stream
    for (int i = 0; i < num_points; i++) {
        classMap[i] = class_stream.read();
    }

    // ----------------------- Computation End ---------------------------
    clock_gettime(CLOCK_MONOTONIC, &end);
    duration = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("\tComputation time: %f seconds\n", duration);

    // Output results for verification
    std::cout << "Final Centroids:" << std::endl;
    for (int i = 0; i < clusters; i++) {
        std::cout << "Centroid [" << i << "]: ";
        for (int d = 0; d < dims; d++) {
            std::cout << centroids[i * dims + d] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "First 10 Class Assignments:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "Point " << i << ": Class = " << classMap[i] << std::endl;
    }

    writeResult(classMap, num_points, argv[6]);
    
    // Free memory
    free(data);
    free(classMap);
    free(centroidPos);
    free(centroids);
    
    return 0;
}