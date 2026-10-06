#include "KMEANS.h"
#include "hls_math.h"
#include <hls_stream.h>
#include <float.h>
#include <math.h>

#define ndims 2         // or pass this as a template parameter
#define npoints 20      // Number of points
#define K 2             // Number of clusters
#define nclusters 2     // Number of clusters
#define minChanges 0
#define maxIterations 10
#define maxThreshold 0

void do_compute(
    hls::stream<float>& data_stream,
    hls::stream<int>& class_stream,
    float* centroids) {

    #pragma HLS INTERFACE axis port=data_stream
    #pragma HLS INTERFACE axis port=class_stream
    #pragma HLS INTERFACE m_axi port=centroids offset=slave bundle=gmem depth=K*ndims
    #pragma HLS INTERFACE s_axilite port=centroids bundle=control
    #pragma HLS INTERFACE s_axilite port=return

    int it = 0;
    int changes = 0;
    float maxDist = FLT_MIN;
    int pointsPerClass[K] = {0};
    float auxCentroids[K * ndims] = {0};
    float distCentroids[K] = {0};
    int classMap[npoints] = {0};

    #pragma HLS array_partition variable=auxCentroids cyclic factor=ndims
    #pragma HLS array_partition variable=pointsPerClass complete
    #pragma HLS array_partition variable=distCentroids complete

    //----------------------------------------------------------------------------

    // 1. Stream in data points
    float local_data[npoints * ndims];
    data_load:
    for (int i = 0; i < npoints; i++) {
        for (int d = 0; d < ndims; d++) {
            #pragma HLS PIPELINE II=1
            local_data[i * ndims + d] = data_stream.read();
        }
    }
    // 2. Stream in centroids
    float local_centroids[K * ndims];
    centroid_load:
    for (int i = 0; i < K * ndims; i++) {
        #pragma HLS PIPELINE II=1
        local_centroids[i] = centroids[i];
    }
    //----------------------------------------------------------------------------

    for (int it = 0; it < maxIterations; it++) {
        #pragma HLS loop_tripcount min=1 max=maxIterations
        
        changes = 0;

        // 1. Calculate the distance from each point to the centroid
        // Assign each point to the nearest centroid.
        S1_points_0: 
        for (int i = 0; i < npoints; i++) {
            #pragma HLS PIPELINE II=1
            
            // Find nearest centroid
            int best_class = 0;
            float minDist = FLT_MAX;
            
            S1_Clusters_1: 
            for (int j = 0; j < K; j++) {
                float dist = 0.0;
                
                S1_Dims_2: 
                for (int d = 0; d < ndims; d++) {
                    #pragma HLS UNROLL
                    float diff = local_data[i * ndims + d] - local_centroids[j * ndims + d];
                    dist += diff * diff;
                }
                
                if (dist < minDist) {
                    minDist = dist;
                    best_class = j + 1;
                }
            }
            if (classMap[i] != best_class) {
                changes++;
                classMap[i] = best_class;
            }
            
            int Class = best_class - 1;
            pointsPerClass[Class]++;
            
            // Move here !!
            S2_Dims_1: 
            for (int d = 0; d < ndims; d++) {
                #pragma HLS UNROLL
                auxCentroids[Class * ndims + d] += local_data[i * ndims + d];
            }
        }

        // 2. Recalculates the centroids: calculates the mean within each cluster
        S2_KDims_0:
        for (int i = 0; i < K; i++) {
            #pragma HLS PIPELINE II=1
            pointsPerClass[i] = 0;
            for (int d = 0; d < ndims; d++) {
                auxCentroids[i * ndims + d] = 0.0;
            }
        }

        S2_Clusters_0: 
        for (int i = 0; i < K; i++) {
            #pragma HLS PIPELINE II=1
            S2_Dims2_1:
            for (int d = 0; d < ndims; d++) {
                #pragma HLS UNROLL
                if (pointsPerClass[i] > 0) {
                    local_centroids[i * ndims + d] = auxCentroids[i * ndims + d] / pointsPerClass[i];
                }
            }
        }

        // 3. check the centroid distance between 2 iterations.
        maxDist = FLT_MIN;
        S3_Clusters_0: 
        for (int i = 0; i < K; i++) {
            #pragma HLS PIPELINE II=1
            distCentroids[i] = 0.0;
            S3_Dims_1:
            for (int d = 0; d < ndims; d++) {
                #pragma HLS UNROLL
                float diff = local_centroids[i * ndims + d] - (auxCentroids[i * ndims + d] / pointsPerClass[i]);
                distCentroids[i] += diff * diff;
            }
            if (distCentroids[i] > maxDist) {
                maxDist = distCentroids[i];
            }
        }
        maxDist = hls::sqrtf(maxDist);

        if (changes <= minChanges || maxDist <= maxThreshold) break;    
    }

    // Stream out results
    output_results:
    for (int i = 0; i < npoints; i++) {
        #pragma HLS PIPELINE II=1
        class_stream.write(classMap[i]);
    }

    // Stream out centroids
    centroid_store:
    for (int i = 0; i < K * ndims; i++) {
        #pragma HLS PIPELINE II=1
        centroids[i] = local_centroids[i];
    }
}