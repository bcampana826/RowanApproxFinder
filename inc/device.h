#ifndef DEVICE_H
#define DEVICE_H

#include "./common.h"

__device__ float get_node_viability_score_device(unsigned int query_node, unsigned int *query_atts, unsigned int *data_atts_orders_offset,
                                          unsigned int *query_signatures);

__device__ unsigned int get_val_from_array_and_buf(unsigned int *array, unsigned int *buffer, unsigned int max, unsigned int index);

__global__ void initialize_searching(G_pointers query_pointers,G_pointers data_pointers, E_pointers extra_pointers);

__global__ void approximate_search_kernel(G_pointers query_pointers,G_pointers data_pointers, E_pointers extra_pointers, unsigned int iter,
                                          unsigned int jobs);

__device__ void construct_partial_path(unsigned int *partial_paths, unsigned int *counter, unsigned int *result_lengths, unsigned int *index_table,
                                       unsigned int *results_table, unsigned int lane_id, unsigned int warp_id, unsigned int iter,
                                       unsigned int pre_idx);

__device__ unsigned int get_score_specific(unsigned int mode, unsigned int score);

__device__ bool check_data_node_query_node_map_score(unsigned int *partial_paths, unsigned int iter, unsigned int data_node,
                                                     G_pointers query_pointers, G_pointers data_pointers, E_pointers extra_pointers);

__device__ void find_new_good_candidates(unsigned int *partial_paths, unsigned int lane_id, unsigned int iter, unsigned int *candidates,
                                         unsigned int *buffer, unsigned int *candidate_count, G_pointers q_p, G_pointers d_p, E_pointers e_p);

__device__ float candidate_score(unsigned int *partial_paths, unsigned int iter, unsigned int data_node,
                                G_pointers q_p, G_pointers d_p, E_pointers e_p);


#endif