#ifndef COMMON_H
#define COMMON_H

#include <string>
#include <vector>
#include <set>
#include <fstream>
#include <iostream>
#include <queue>
#include <chrono>
#include "omp.h"
#include "cuda.h"
#include "cuda_runtime.h"

#define Signature_Properties 2
#define In_degree_offset 0
#define Out_degree_offset 1
#define BLK_NUMS 1
#define BLK_DIM 128
#define GPU_TABLE_SIZES 10000
#define WEIGHT_MISSING_EDGE .25
#define WEIGHT_MISSING_VERT .5
#define WEIGHT_INTRA_VERT 1.0
#define MIN_FIND_SIM_SCORE 0.5
#define MAX_QUERY_NODES 12
#define MAX_EDGES 112
#define WARPS_EACH_BLK (BLK_DIM / 32)
#define BUFFER_PER_WARP 100000000
#define BUFFER_TABLE_SIZE (BUFFER_PER_WARP * WARPS_EACH_BLK * BLK_NUMS)

using namespace std;

typedef struct G_pointers
{
    unsigned int *outgoing_neighbors;
    unsigned int *outgoing_neighbors_offset;
    unsigned int *signatures;
    unsigned int *incoming_neighbors;
    unsigned int *incoming_neighbors_offset;
    unsigned int *attributes;
    unsigned int *attributes_in_order;
    unsigned int *attributes_in_order_offset;
    unsigned int *all_neighbors;
    unsigned int *all_neighbors_offset;
    unsigned int V;
    unsigned int E;
    unsigned int num_attributes;
} G_pointers; // graph related

typedef struct E_pointers
{
    unsigned int *matching_order;
    unsigned int *global_count;
    unsigned int *result_lengths;
    unsigned int *results_table;
    unsigned int *indexes_table;
    unsigned int *scores_table;
    unsigned int *intra_v_table;
    unsigned long long int *write_pos;
    unsigned int *helper_buffer;
} E_pointers;

#endif