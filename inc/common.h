#ifndef COMMON_H
#define COMMON_H

#include <string>
#include <vector>
#include <set>
#include <fstream>
#include <iostream>
#include <queue>
#include "omp.h"
#include "cuda.h"
#include "cuda_runtime.h"

#define Signature_Properties 2
#define In_degree_offset 0
#define Out_degree_offset 1
#define BLK_NUMS 108
#define BLK_DIM 512
#define GPU_TABLE_SIZES 5000
#define WEIGHT_MISSING_EDGE 1
#define WEIGHT_MISSING_VERT 1
#define WEIGHT_INTRA_VERT 1
#define MID_FIND_SIM_SCORE 0.5
#define MAX_QUERY_NODES 12
#define MAX_EDGES 112
#define WARPS_EACH_BLK (BLK_DIM/32)

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
    unsigned int *two_hop_neighbors_offset;
    unsigned int *two_hop_neighbors;
    unsigned int V;
    unsigned int E;
    unsigned int num_attributes;
} G_pointers; // graph related

typedef struct E_pointers
{
    unsigned int *matching_order;
    unsigned int *result_lengths;
    unsigned int *results_table;
    unsigned int *indexes_table;
    unsigned int *scores_table;
    unsigned int *intra_v_table;
    unsigned long long int *write_pos;
} E_pointers;

#endif