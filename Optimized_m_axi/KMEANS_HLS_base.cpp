// top level function or kernel function implementation

#include "KMEANS.h"
#include "hls_math.h"
#include <float.h>
#include <math.h>

#define ndims 2         // or pass this as a template parameter
#define npoints 20       // Number of points
#define K 2             // Numeber of clusters
#define nclusters 2     // Numeber of clusters
#define minChanges 0
#define maxIterations 10
#define maxThreshold 0

/**
 * Function euclideanDistanceSquared: square of the Euclidean distance
 * we always compare the square of euclidean distance to avoid sqrt function calls
 */
// float euclideanDistanceSquared(float *point, float *center, int ndims) {
//     float dist = 0.0;

//     for (int i = 0; i < ndims; i++) {
//         dist += (point[i] - center[i]) * (point[i] - center[i]);
//     }

//     return dist;
// }

/**
 * do_compute: the entry of your top level function or kernel function that implements the K-Means algorithm
 * This is an almost complete copy of the sequential function except for the timing
 */
void do_compute(
    float data[npoints * ndims], 
    int classMap[npoints], 
    float centroids[K * ndims]) {

    #pragma HLS INTERFACE m_axi port=data       offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=classMap   offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=centroids  offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=data       bundle=control
    #pragma HLS INTERFACE s_axilite port=classMap   bundle=control
    #pragma HLS INTERFACE s_axilite port=centroids  bundle=control
    #pragma HLS INTERFACE s_axilite port=return

    int it = 0;
    int changes = 0;
    float maxDist = FLT_MIN;
    int pointsPerClass[K];
    float distCentroids[K];

    float auxCentroids [K * ndims];
    #pragma HLS array_partition variable=auxCentroids cyclic factor=ndims dim=1
    
    //----------------------------------------------------------------------------

    float local_centroids[K * ndims];
    #pragma HLS array_partition variable=local_centroids cyclic factor=K*ndims dim=1
    centroid_buf:
    for (int i =0; i < K * ndims; i ++) {
        #pragma HLS PIPELINE II=1
        local_centroids[i] = centroids[i];
    }
    
    //----------------------------------------------------------------------------

    float local_data[npoints * ndims];
    #pragma HLS array_partition variable=local_data cyclic factor=npoints*ndims dim=1
    data_buf:
    for (int i =0; i < npoints * ndims; i ++) {
        #pragma HLS PIPELINE II=1
        local_data[i] = data[i];
    }

    //----------------------------------------------------------------------------

    S0_Iteration:
    for (int it = 0; it < maxIterations; it++) {
        #pragma HLS loop_tripcount min=1 max=maxIterations
        // 1. Calculate the distance from each point to the centroid
        // Assign each point to the nearest centroid.
        changes = 0;

        S1_points_0: 
        for (int i = 0; i < npoints; i++) {
            #pragma HLS PIPELINE II=1
            int Class = 1;
            float minDist = FLT_MAX;
            S1_Clusters_1:
            for (int j = 0; j < K; j++) {
                float dist = 0.0;

                S1_Dims_2:
                for (int d = 0; d < ndims; d++) {
                    #pragma HLS unroll
                    dist += (local_data[(i * ndims) + d] - local_centroids[(j * ndims) + d]) * (local_data[(i * ndims) + d] - local_centroids[(j * ndims) + d]);
                }
                if (dist < minDist) {
                    minDist = dist;
                    // Note that the centroid class id starts from 1
                    // centroid id = 0 indicates that the point is not assigned to a cluster yet
                    Class = j + 1;
                }
            }
            if (classMap[i] != Class) {
                changes++;
            }
            classMap[i] = Class;
        }

        // 2. Recalculates the centroids: calculates the mean within each cluster
        memset(pointsPerClass, 0, K * sizeof(int));
        S2_KDims_0: 
        for (int i = 0; i < K * ndims; i++) {
            #pragma HLS pipeline II=1
            auxCentroids[i] = 0.0;
        }
        
        S2_Points_0: 
        for (int i = 0; i < npoints; i++) {
            #pragma HLS pipeline II=1
            int Class = classMap[i];
            pointsPerClass[Class - 1] = pointsPerClass[Class - 1] + 1;
            S2_Dims_1: 
            for (int j = 0; j < ndims; j++) {
                #pragma HLS unroll
                auxCentroids[(Class - 1) * ndims + j] += local_data[i * ndims + j];
            }
        }
        
        S2_Clusters_0: 
        for (int i = 0; i < K; i++) {
            #pragma HLS pipeline II=1
            S2_Dims2_1: 
            for (int j = 0; j < ndims; j++) {
                #pragma HLS unroll
                auxCentroids[i * ndims + j] /= pointsPerClass[i];
            }
        }

        // 3. check the centroid distance between 2 iterations.
        maxDist = FLT_MIN;

        S3_Clusters_0: 
        for (int i = 0; i < K; i++) {
            #pragma HLS pipeline II=1
            distCentroids[i] = 0.0;

                S3_Dims_1: 
                for (int d = 0; d < ndims; d++) {
                #pragma HLS UNROLL  // Fully unroll
                    distCentroids[i] += (local_centroids[(i * ndims) + d] - auxCentroids[(i * ndims) + d]) * (local_centroids[(i * ndims) + d] - auxCentroids[(i * ndims) + d]);
                }
            if (distCentroids[i] > maxDist) {
                maxDist = distCentroids[i];
            }
        }

        maxDist = hls::sqrtf(maxDist);

        memcp_1: 
        for (int i = 0; i < K; i++) {
            #pragma HLS pipeline II=1
            memcp_2: 
            for (int d = 0; d < ndims; d++) {
                #pragma HLS unroll
                local_centroids[(i * ndims) + d] = auxCentroids[(i * ndims) + d];
            }
        }
    if (changes <= minChanges || maxDist <= maxThreshold) break;    
    }
}
