#include "../inc/device.h"

/**
 * this is the score
 * |C(node)| / degree   
 * 
 * it only considers out neighbors for degree
*/
__device__ float get_node_viability_score(unsigned int node, unsigned int *atts, unsigned int *atts_orders_offset,
                                          unsigned int *signatures){
    unsigned int attribute = atts[node];
    unsigned int canidates = atts_orders_offset[attribute+1] - atts_orders_offset[attribute];

    if( signatures[node*Signature_Properties+1] != 0 ){
        return static_cast< float >( canidates ) / signatures[node*Signature_Properties+1];
    }
    else{
        return 0;
    }
}

/**
 * This method is going to set the root node and the matching order for our programs use later
*/
__device__ float root_and_matching_order(){

}