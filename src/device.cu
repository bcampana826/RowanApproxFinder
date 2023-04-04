#include "../inc/device.h"

/**
 * this is the score
 * |C(node)| / degree   
 * 
 * it only considers out neighbors for degree
*/
__device__ float get_node_viability_score_device(unsigned int query_node, unsigned int *query_atts, unsigned int *data_atts_orders_offset,
                                          unsigned int *query_signatures){

    // grab the query nodes attribute
    unsigned int attribute = query_atts[query_node];

    // grab the number of datanodes with that attribute
    unsigned int canidates = data_atts_orders_offset[attribute+1] - data_atts_orders_offset[attribute];

    // if it has more than 0 neighbors
    if( (query_signatures[query_node*Signature_Properties+1] + query_signatures[query_node*Signature_Properties+0]) != 0 ){

        // return this math
        return static_cast< float >( canidates ) / (query_signatures[query_node*Signature_Properties+1] + query_signatures[query_node*Signature_Properties+0]);
    }
    else{
        return 0.0;
    }
}

/**
 * 
 * 
 * NOT DONE
 * 
 * 
*/

// Generates the score for given index in the results table.
__device__ float get_score(G_pointers query_pointers,G_pointers data_pointers,E_pointers extra_pointers, unsigned int index, unsigned int iter){

    // W1 * ME + W2 * MV + W3 * IV

    
    
    
    return 0.0;
}


/**
 * 
 * 
 * done
 * 
 * 
*/

__global__ void initialize_searching(G_pointers query_pointers,G_pointers data_pointers,E_pointers extra_pointers){
        
    extra_pointers.result_lengths[0] = 0;
    // Initialize the first row of the results table
    unsigned int v = extra_pointers.matching_order[0];

    unsigned int att = query_pointers.attributes[v];

    unsigned int candidates =  data_pointers.attributes_in_order_offset[att+1] - data_pointers.attributes_in_order_offset[att];

    unsigned int globalIndex = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int totalThreads = BLK_NUMS*blockDim.x;

    unsigned int start = data_pointers.attributes_in_order_offset[att];

    // for each node that shares attribute with mo[0], add it to the results table
    // parrallel, so skip for each thread
    for(unsigned int i = globalIndex; i < candidates; i += totalThreads){

            unsigned int data = data_pointers.attributes_in_order[i+start];

            unsigned int index = atomicAdd(&extra_pointers.result_lengths[1],1);
            //extra_pointers.indexes_table[index] = -1;
            extra_pointers.results_table[index] = data;
            extra_pointers.intra_v_table[index] = 0;
            
            // so for others we will have a get score that takes the precopy to do this. For now, were just doing that formula here manually
            extra_pointers.scores_table[index] = (query_pointers.V-1)*WEIGHT_MISSING_VERT+(query_pointers.E)*WEIGHT_MISSING_EDGE;
    }

}

__device__ void candidate_sim_score(unsigned int *partial_paths, unsigned int node_to_check, unsigned int iter, unsigned int *data_out_neighbors,
                                    unsigned int *data_in_neighbors, unsigned int *data_in_offset, unsigned int *data_out_offset){

                                    }

__device__ void construct_partial_path(unsigned int *partial_paths, unsigned int *counter, unsigned int *result_lengths, unsigned int *index_table,
                                       unsigned int *results_table, unsigned int lane_id, unsigned int warp_id, unsigned int iter, 
                                       unsigned int pre_idx){

    // first thread for each warp only
    if(lane_id==0){
        unsigned int vertexLocation;
        counter[warp_id] = 0;

        for(int i = iter - 1; i >= 0; i--){

            if(i == iter - 1){
                vertexLocation = result_lengths[i] + pre_idx;
            } else {
                
                vertexLocation = index_table[vertexLocation];
            }

            partial_paths[warp_id * MAX_QUERY_NODES + i] = results_table[vertexLocation];


        }
    }
}

__device__ void check_data_node_query_node_map_score(unsigned int *partial_paths, unsigned int iter, unsigned int data_node,
                                                    G_pointers query_pointers,G_pointers data_pointers, E_pointers extra_pointers){

    unsigned int * mapped_neighbors = new unsigned int[iter];

    unsigned int incoming = data_pointers.incoming_neighbors_offset[data_node+1] - data_pointers.incoming_neighbors_offset[data_node];
    unsigned int outgoing = data_pointers.outgoing_neighbors_offset[data_node+1] - data_pointers.outgoing_neighbors_offset[data_node];

    unsigned int max = incoming > outgoing ? incoming : outgoing;

    
    for(unsigned int i = 0; i < iter; i++){

        for(unsigned int j = 0; j < max; j++ ){
            if(j<incoming){
                if (data_pointers.incoming_neighbors[data_pointers.outgoing_neighbors_offset[data_node]+j] == partial_paths[i]){
                    mapped_neighbors[i] = partial_paths[i];
                    break;
                }
            }

            if(j<outgoing){
                if (data_pointers.outgoing_neighbors[data_pointers.outgoing_neighbors_offset[data_node]+j] == partial_paths[i]){
                    mapped_neighbors[i] = partial_paths[i];
                    break;
                }
            }
        

        }

    }


}

/**
 * Grab each neighbor X-Hops away from all pre-copy nodes
 * 
 * 
 * 
 * 
*/
__device__ void find_good_canidates(unsigned int *partial_paths,unsigned int lane_id, unsigned int iter, unsigned int * candidates, 
                                    unsigned int * buffer, unsigned int * temp_neighbors, G_pointers query_pointers,G_pointers data_pointers, 
                                    E_pointers extra_pointers ){

    unsigned int count = 0;
    unsigned int k_hop = 2;

    unsigned int query_att = query_pointers.attributes[extra_pointers.matching_order[iter]];

    // pre init partial_paths into temp_neighbors
    for(int i = threadIdx.x; i < iter; i+=32){
        temp_neighbors[i] = partial_paths[i];
    }

    unsigned int jobs = iter;

    while(count < k_hop){

        // first, go through all the pre copy neighbors.  Find potential canididates and add to candidates in shared mem
        // note, some threads here will be waiting.
        for(int i = threadIdx.x; i < jobs; i+=32){

            unsigned int start = data_pointer.outgoing_neighbors_offset[partial_paths[i]];
            unsigned int end = data_pointer.outgoing_neighbors_offset[partial_paths[i]+1];

            for(int j = start; j < end; j++){
                // these are one hop matches

                if (data_pointers.attributes[data_pointer.outgoing_neighbors[start+j]] == query_att){
                    // check if good candidate






                }
                
                // if good candidate
                    // if not in added set
                        // add to thread_cands
                            // combine thread_cands

                

            }


        }


    }

    









}


__global__ void approximate_search_kernel(G_pointers query_pointers,G_pointers data_pointers, E_pointers extra_pointers,
                                          unsigned int iter, unsigned int jobs, unsigned int *global_count){

    // this is one of the lists used in the cuTS implementation.
    __shared__ unsigned int partial_paths[WARPS_EACH_BLK*MAX_QUERY_NODES]; // 24,576 ( 3.072 kb )
    __shared__ unsigned int counter[WARPS_EACH_BLK]; // 48 

    __shared__ unsigned int temp_neighbors[WARPS_EACH_BLK*MAX_EDGES]; // 229,376 ( 28.672 kb )
    __shared__ unsigned int canidates[WARPS_EACH_BLK*MAX_EDGES]; // 229,376 ( 28.672 kb )

    // total shared mem used = 60.422 kilobytes per block

    unsigned int v = extra_pointers.matching_order[iter];
    unsigned int mask = 0xFFFFFFFF;
    unsigned int warp_id = threadIdx.x/32;
    unsigned int lane_id = threadIdx.x%32;
    unsigned int global_idx = (blockIdx.x)*WARPS_EACH_BLK+warp_id;

    while(true){

        unsigned int pre_idx;
        
        //first thread in each warp (  threadIdx.x%32; )
        if(lane_id == 0){

            pre_idx = atomicAdd(&global_count[0],1);
        }

        pre_idx = __shfl_sync(mask,pre_idx,0);
        if(pre_idx>=jobs){
            break ;
        }
 

        // construct partial paths
        construct_partial_path(partial_paths, counter, extra_pointers.result_lengths, extra_pointers.indexes_table, extra_pointers.results_table,
                                lane_id,warp_id,iter,pre_idx);

        find_good_canidates(&partial_paths[warp_id * MAX_QUERY_NODES],G_pointers query_pointers,G_pointers data_pointers)


    }





}

