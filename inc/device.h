#ifndef DEVICE_H
#define DEVICE_H

#include "./common.h"

__device__ float get_node_viability_score_device(unsigned int query_node, unsigned int *query_atts, unsigned int *data_atts_orders_offset,
                                          unsigned int *query_signatures);

__device__ float get_score(G_pointers query_pointers,G_pointers data_pointers,E_pointers extra_pointers, unsigned int index, unsigned int iter);

__global__ void initialize_searching(unsigned int *result_lengths);

__global__ void initialize_searching(G_pointers query_pointers,G_pointers data_pointers, E_pointers extra_pointers, unsigned int *result_lengths);

#endif