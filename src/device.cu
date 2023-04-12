#include "../inc/device.h"

/**
 * this is the score
 * |C(node)| / degree
 *
 * it only considers out neighbors for degree
 */
__device__ float get_node_viability_score_device(unsigned int query_node, unsigned int *query_atts, unsigned int *data_atts_orders_offset,
                                                 unsigned int *query_signatures)
{

    // grab the query nodes attribute
    unsigned int attribute = query_atts[query_node];

    // grab the number of datanodes with that attribute
    unsigned int canidates = data_atts_orders_offset[attribute + 1] - data_atts_orders_offset[attribute];

    // if it has more than 0 neighbors
    if ((query_signatures[query_node * Signature_Properties + 1] + query_signatures[query_node * Signature_Properties + 0]) != 0)
    {

        // return this math
        return static_cast<float>(canidates) / (query_signatures[query_node * Signature_Properties + 1] + query_signatures[query_node * Signature_Properties + 0]);
    }
    else
    {
        return 0.0;
    }
}

__global__ void initialize_searching(G_pointers query_pointers, G_pointers data_pointers, E_pointers extra_pointers)
{

    extra_pointers.result_lengths[0] = 0;
    // Initialize the first row of the results table
    unsigned int v = extra_pointers.matching_order[0];

    unsigned int att = query_pointers.attributes[v];

    unsigned int candidates = data_pointers.attributes_in_order_offset[att + 1] - data_pointers.attributes_in_order_offset[att];

    unsigned int globalIndex = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int totalThreads = BLK_NUMS * blockDim.x;

    unsigned int start = data_pointers.attributes_in_order_offset[att];

    // for each node that shares attribute with mo[0], add it to the results table
    // parrallel, so skip for each thread
    for (unsigned int i = globalIndex; i < candidates; i += totalThreads)
    {
        printf("(%d)", i);
        unsigned int data = data_pointers.attributes_in_order[i + start];

        unsigned int index = atomicAdd(&extra_pointers.result_lengths[1], 1);
        // extra_pointers.indexes_table[index] = -1;
        printf("{%d}", index);
        printf("[%d]", data);
        extra_pointers.results_table[index] = data;
        extra_pointers.intra_v_table[index] = 0;

        // so for others we will have a get score that takes the precopy to do this. For now, were just doing that formula here manually
        extra_pointers.scores_table[index] = (query_pointers.V - 1) * WEIGHT_MISSING_VERT + (query_pointers.E) * WEIGHT_MISSING_EDGE;
    }
}

__device__ void construct_partial_path(unsigned int *partial_paths, unsigned int *counter, unsigned int *result_lengths, unsigned int *index_table,
                                       unsigned int *results_table, unsigned int lane_id, unsigned int warp_id, unsigned int iter,
                                       unsigned int pre_idx)
{

    // first thread for each warp only
    if (lane_id == 0)
    {
        unsigned int vertexLocation;
        counter[warp_id] = 0;

        for (int i = iter - 1; i >= 0; i--)
        {

            if (i == iter - 1)
            {
                vertexLocation = result_lengths[i] + pre_idx;
            }
            else
            {

                vertexLocation = index_table[vertexLocation];
            }

            partial_paths[warp_id * MAX_QUERY_NODES + i] = results_table[vertexLocation];
        }
    }
}

__device__ bool check_if_found(unsigned int *array, unsigned int start, unsigned int end, unsigned int value)
{
    for (int i = start; i < end; i++)
    {
        if (array[i] == value)
        {
            return true;
        }
    }
    return false;
}

__device__ bool check_data_node_query_node_map_score(unsigned int *partial_paths, unsigned int iter, unsigned int data_node,
                                                     G_pointers query_pointers, G_pointers data_pointers, E_pointers extra_pointers)
{

    unsigned int incoming_set = new unsigned int[iter + 1];
    unsigned int outgoing_set = new unsigned int[iter + 1];

    unsigned int inc_count = 0;
    unsigned int out_count = 0;

    unsigned int incoming;
    unsigned int outgoing;
    unsigned int max;

    for (int i = 0; i < iter; i++)
    {
        // method helper variables
        incoming = query_pointers.incoming_neighbors_offset[extra_pointers.matching_order[i] + 1] - query_pointers.incoming_neighbors_offset[extra_pointers.matching_order[i]];
        outgoing = query_pointers.outgoing_neighbors_offset[extra_pointers.matching_order[i] + 1] - query_pointers.outgoing_neighbors_offset[extra_pointers.matching_order[i]];
        max = incoming > outgoing ? incoming : outgoing;

        for (unsigned int j = 0; j < max; j++)
        {
            if (j < incoming)
            {
                if (query_pointers.incoming_neighbors[query_pointers.incoming_neighbors_offset[extra_pointers.matching_order[i]] + j] == extra_pointers.matching_order[iter])
                {
                    if (!check_if_found(incoming_set, 0, inc_count, matching_order[i]))
                    {
                        incoming_set[inc_count] = matching_order[i];
                        inc_count++;
                    }
                }
            }
            if (j < outgoing)
            {
                if (query_pointers.outgoing_neighbors[query_pointers.outgoing_neighbors_offset[extra_pointers.matching_order[i]] + j] == extra_pointers.matching_order[iter])
                {
                    if (!check_if_found(outgoing_set, 0, out_count, matching_order[i]))
                    {
                        outgoing_set[out_count] = matching_order[i];
                        out_count++;
                    }
                }
            }
        }

        // mehtod helper variables
        incoming = data_pointers.incoming_neighbors_offset[data_node + 1] - data_pointers.incoming_neighbors_offset[data_node];
        outgoing = data_pointers.outgoing_neighbors_offset[data_node + 1] - data_pointers.outgoing_neighbors_offset[data_node];
        max = incoming > outgoing ? incoming : outgoing;

        unsigned int inc_found = 0;
        unsigned int out_found = 0;

        for (unsigned int i = 0; i < max; i++)
        {
            if (j < incoming)
            {
                if (check_if_found(incoming_set, 0, inc_count, data_pointers.incoming_neighbors[data_pointers.incoming_neighbors_offset[data_node] + i]))
                {
                    inc_found++;
                }
            }
            if (j < outgoing)
            {
                if (check_if_found(outgoing_set, 0, out_count, data_pointers.incoming_neighbors[data_pointers.outgoing_neighbors_offset[data_node] + i]))
                {
                    out_found++;
                }
            }
        }

        float score = (static_cast<float>(inc_found) + (out_found)) / (inc_count + out_count);

        return score > MIN_FIND_SIM_SCORE;
    }
}

/**
 * As 0f 4/12, completing this only checking next door neighbors
 *
 * Method is as follows
 *
 * 1. Save the partial paths into temp_nieghbors for this warp, Then Sync
 * 2. Now, each node enters a while loop where it will perform checks for each neighbor
 *
 *
 *
 */
__device__ void find_good_canidates(unsigned int *partial_paths, unsigned int lane_id, unsigned int iter, unsigned int *candidates,
                                    unsigned int *buffer, unsigned int *temp_neighbors, unsigned int candidate_count, G_pointers query_pointers,
                                    G_pointers data_pointers, E_pointers extra_pointers)
{

    unsigned int count = 0;
    unsigned int k_hop = 1;
    unsigned int jobs = iter;
    unsigned int query_att = query_pointers.attributes[extra_pointers.matching_order[iter]];

    if (lane_id == 0)
    {
        // pre init partial_paths into temp_neighbors
        for (int i = 0; i < iter; i++)
        {
            temp_neighbors[i] = partial_paths[i];
        }
    }
    // sync the warp up now with temp_neighbors setup
    __syncwarp();

    // this while loop is redundant while k_hops = 1.
    while (count < k_hop)
    {

        // first, go through all the pre copy neighbors.  Find potential canididates and add to candidates in shared mem
        // note, some threads here will be waiting.
        unsigned int added_id = threadIdx.x;
        while (true)
        {
            /* code */
            unsigned int node;
            unsigned int neighbor;
            unsigned int sum = 0;
            bool working = false;

            /**
             * This for loop is used to determine which neighbor node we will be checking for given thread.
             */
            for (unsigned int j = 0; j < jobs; j++)
            {

                if (sum + (data_pointer.outgoing_neighbors_offset[temp_neighbors[j] + 1] - data_pointer.outgoing_neighbors_offset[temp_neighbors[j]]) >= added_id)
                {
                    node = j;
                    neighbor = added_id - sum;
                    working = true;
                    break;
                }
                else
                {
                    sum += (data_pointer.outgoing_neighbors_offset[temp_neighbors[j] + 1] - data_pointer.outgoing_neighbors_offset[temp_neighbors[j]]);
                }
            }

            // if this thread doesnt have a neighbor to check, break out.
            if (!working)
            {
                break;
            }

            // Grab that node, check if it shares the attribute, then check its score, then add.

            unsigned int start = data_pointer.outgoing_neighbors_offset[node];
            if (data_pointers.attributes[data_pointer.outgoing_neighbors[start + neighbor]] == query_att)
            {
                if (check_data_node_query_node_map_score(partial_paths, iter, data_pointers.attributes[data_pointer.outgoing_neighbors[start + j]],
                                                         query_pointers, data_pointers, extra_pointers))
                {
                    // found a candidate
                    unsigned int idx = atomicAdd(&candidate_count, 1) - 1;
                    if (idx < MAX_EDGES)
                    {
                        // add to canidate array
                        candidates[idx] = data_pointer.outgoing_neighbors[start + neighbor];
                    }
                    else
                    {
                        // add to helper buffer instead
                        buffer[idx - MAX_EDGES] = data_pointer.outgoing_neighbors[start + neighbor];
                    }
                }
            }
        }

        // count would be increased here
        __syncwarp();
    }
}

__device__ unsigned int get_val_from_array_and_buf(unsigned int *array, unsigned int *buffer, unsigned int max, unsigned int index)
{
    if (index > max)
    {
        return buffer[index - max];
    }
    else
    {
        return array[index];
    }
}

__device__ void sort_out_dupes(unsigned int *candidates, unsigned int *buffer, unsigned int *count, unsigned int lane_id)
{

    unsigned int new_count = 0;
    bool dupe;
    // has to be done by 1 thread
    if (lane_id == 0)
    {
        for (int i = 0; i < count[0]; i++)
        {

            unsigned int checking = get_val_from_array_and_buf(candidates, buffer, MAX_EDGES, i);
            dupe = false;
            for (int j = 0; j < i; j++)
            {
                if (checking == get_val_from_array_and_buf(candidates, buffer, MAX_EDGES, j))
                {
                    // duplicate, do not add
                    dupe = true;
                    break;
                }
            }

            if (!dupe)
            {
                if (new_count < MAX_EDGES)
                {
                    candidates[new_count] = checking;
                }
                else
                {
                    buffer[new_count - MAX_EDGES] = checking;
                }
                new_count++;
            }
        }

        count[0] = new_count++;
    }
    __syncwarp();
}

__device__ unsigned int get_graph_score(unsigned int *partial_paths, unsigned int iteration, unsigned int score_candidate,
                                        G_pointers query_pointers, G_pointers data_pointers, E_pointers extra_pointers)
{
    unsigned int unq_vertexes = 0;
    unsigned int correct_inc_edge_count = 0;
    unsigned int correct_out_edge_count = 0;

    unsigned int start;
    unsigned int end;
    unsigned int order_node;

    for (unsigned int i = 0; i < iteration; i++)
    {
        // comparing partial paths [ i ] to matching order [ i ]
        if (!check_if_found(partial_paths, 0, i, partial_paths[i]))
        {
            unq_vertexes++;

            order_node = extra_pointers.matching_order[i];
            start = query_pointers.incoming_neighbor_offset[order_node];
            end = query_pointers.incoming_neighbor_offset[order_node + 1];

            for (unsigned int j = start; j < end; j++)
            {
            }
        }
    }
}

__global__ void approximate_search_kernel(G_pointers query_pointers, G_pointers data_pointers, E_pointers extra_pointers,
                                          unsigned int iter, unsigned int jobs, unsigned int *global_count)
{

    // this is one of the lists used in the cuTS implementation.
    __shared__ unsigned int partial_paths[WARPS_EACH_BLK * MAX_QUERY_NODES]; // 24,576 ( 3.072 kb )

    __shared__ unsigned int cand_counter[WARPS_EACH_BLK]; // 48

    __shared__ unsigned int temp_neighbors[WARPS_EACH_BLK * MAX_EDGES]; // 229,376 ( 28.672 kb )
    __shared__ unsigned int canidates[WARPS_EACH_BLK * MAX_EDGES];      // 229,376 ( 28.672 kb )

    // total shared mem used = 60.422 kilobytes per block

    unsigned int v = extra_pointers.matching_order[iter];
    unsigned int mask = 0xFFFFFFFF;
    unsigned int warp_id = threadIdx.x / 32;
    unsigned int lane_id = threadIdx.x % 32;
    unsigned int global_idx = (blockIdx.x) * WARPS_EACH_BLK + warp_id;

    while (true)
    {

        unsigned int pre_idx;

        // first thread in each warp (  threadIdx.x%32; )
        if (lane_id == 0)
        {

            pre_idx = atomicAdd(&global_count[0], 1);
        }

        pre_idx = __shfl_sync(mask, pre_idx, 0);
        if (pre_idx >= jobs)
        {
            break;
        }

        // construct partial paths
        construct_partial_path(partial_paths, counter, extra_pointers.result_lengths, extra_pointers.indexes_table, extra_pointers.results_table,
                               lane_id, warp_id, iter, pre_idx);

        find_good_canidates(&partial_paths[warp_id * MAX_QUERY_NODES], lane_id, iter, &candidates[warp_id * MAX_EDGES],
                            &extra_pointers.helper_buffer[global_idx * BUFFER_PER_WARP], &temp_neighbors[warp_id * MAX_EDGES], &cand_counter[warp_id],
                            query_pointers, data_pointers, extra_pointers);

        // we need to sort out duplicate candidates
        sort_out_dupes(&candidates[warp_id * MAX_EDGES], &extra_pointers.helper_buffer[global_idx * BUFFER_PER_WARP], &cand_counter[warp_id], lane_id);

        // now, we can write out the values

        // IF this is the final iteration
        if (iter == query_pointers.V - 1)
        {
            unsigned long long int write_pos;
            if (lane_id == 0)
            {
                // add the intersection_len+iteration number to s_p.write_pos and write pos
                write_pos = atomicAdd(&extra_pointers.write_pos[0], intersection_len + iter);

                unsigned long long int index_pos = atomicAdd(&s_p.indexes_pos[0], 1);
                index_pos = 2 * index_pos;

                // we dont want to overwrite things
                s_p.final_results_row_ptrs[index_pos] = write_pos;
                s_p.final_results_row_ptrs[index_pos + 1] = write_pos + intersection_len + iter;
            }
            // only the second one does this
            if (lane_id == 1)
            {
                atomicAdd(&s_p.final_count[0], intersection_len);
            }
            write_pos = __shfl_sync(0xFFFFFFFF, write_pos, 0);
            if (lane_id < 5)
            {
                for (unsigned int i = lane_id; i < iter; i += 5)
                {
                    s_p.final_results_table[write_pos + i] = pre_copy[warp_id * QUERY_NODES + i];
                }
            }
            else
            {
                unsigned int j = lane_id - 5;
                for (unsigned int i = j; i < intersection_len; i += 27)
                {
                    unsigned int candidate;
                    if (i < MAX_NE)
                    {
                        candidate = intersection_result[warp_id * MAX_NE + i];
                    }
                    else
                    {
                        candidate = intersection_helper_result[helperOffset + i - MAX_NE];
                    }
                    s_p.final_results_table[write_pos + i + iter] = candidate;
                }
            }
        }
        else
        {
            // This level has been completed, we need to write it
            unsigned int write_offset;

            // for each found candidate
            for (unsigned int i = lane_id; i < cand_counter[warp_id]; i += 32)
            {
                unsigned int candidate;
                if (i < MAX_EDGES)
                {
                    candidate = candidates[warp_id * MAX_EDGES + i];
                }
                else
                {
                    candidate = extra_pointers.helper_buffer[(BUFFER_PER_WARP * warp_id) + i - MAX_NE];
                }

                // increase the offset and save them in the results and index table
                write_offset = atomicAdd(&extra_pointers.result_lengths[iter + 1], 1);
                extra_pointers.results_table[write_offset] = candidate;
                extra_pointers.indexes_table[write_offset] = pre_idx + extra_pointers.lengths[iter - 1];
                // unused atm
                // extra_pointers.intra_v_table
                extra_pointers.scores_table[write_offset] = get_graph_score(&partial_paths[warp_id * MAX_QUERY_NODES], iter, candidate)
            }
        }
    }
}
