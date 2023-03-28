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

__global__ void initialize_searching(unsigned int *result_lengths){
    //unsigned int index = atomicAdd(&result_lengths[0],1);
    printf("calling kernel\n");
}

/**
 * 
 * 
 * NOT DONE
 * 
 * 
*/

__global__ void initialize_searching(G_pointers query_pointers,G_pointers data_pointers,E_pointers extra_pointers, unsigned int *result_lengths){
        
    //extra_pointers.result_lengths[0] = 0;
    printf("calling kernel\n");
    // Initialize the first row of the results table
    unsigned int v = extra_pointers.matching_order[0];
    result_lengths[0] = 5;
    unsigned int att = query_pointers.attributes[v];

    unsigned int candidates =  data_pointers.attributes_in_order_offset[att+1] - data_pointers.attributes_in_order_offset[att];

    unsigned int globalIndex = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int totalThreads = BLK_NUMS*blockDim.x;

    unsigned int start = data_pointers.attributes_in_order_offset[att];

    // for each node that shares attribute with mo[0], add it to the results table
    // parrallel, so skip for each thread
    for(unsigned int i = globalIndex; i < candidates; i += totalThreads){

            unsigned int data = data_pointers.attributes_in_order[i+start];

            unsigned int index = atomicAdd(&result_lengths[1],1);
            //extra_pointers.indexes_table[index] = -1;
            extra_pointers.results_table[index] = data;
            extra_pointers.intra_v_table[index] = 0;
            
            // so for others we will have a get score that takes the precopy to do this. For now, were just doing that formula here manually
            extra_pointers.scores_table[index] = (query_pointers.V-1)*WEIGHT_MISSING_VERT+(query_pointers.E)*WEIGHT_MISSING_EDGE;
    }

}
