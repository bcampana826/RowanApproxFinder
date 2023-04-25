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
            for (int j = 0; j < new_count; j++)
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
                    printf("Buffer\n");
                    buffer[new_count - MAX_EDGES] = checking;
                }
                new_count++;
            }
        }

        count[0] = new_count;
    }
    __syncwarp();
}

__device__ int check_if_found(unsigned int *array, unsigned int start, unsigned int end, unsigned int value)
{
    for (int i = start; i < end; i++)
    {
        if (array[i] == value)
        {
            return i;
        }
    }
    return -1;
}

__device__ unsigned int get_val_from_array_and_buf(unsigned int *array, unsigned int *buffer, unsigned int max, unsigned int index)
{
    if (index >= max)
    {
        //printf("TIME TO PUT IN THE BUFFER \n");
        printf("Buffer\n");
        return buffer[index - max];
    }
    else
    {
        return array[index];
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
        unsigned int data = data_pointers.attributes_in_order[i + start];

        unsigned int index = atomicAdd(&extra_pointers.result_lengths[1], 1);
        // extra_pointers.indexes_table[index] = -1;
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
                //printf("%d,",results_table[vertexLocation]);
            }
            else
            {

                vertexLocation = index_table[vertexLocation];
            }

            partial_paths[warp_id * MAX_QUERY_NODES + i] = results_table[vertexLocation];
            
        }
    }
}

__device__ float candidate_score(unsigned int *partial_paths, unsigned int iter, unsigned int data_node,
                                G_pointers q_p, G_pointers d_p, E_pointers e_p){

    // twice - once for outgoing, once for incoming

    // for each neighbor this query node has - (matching order [iter])
    //      if that neighbor is in the partial path
    //          add 1 to total
    //          if data_node has that partial path node in its neighbor set
    //              add 1 to mapped

    // score = mapped/total
    //printf("\npartial path: %d, node: %d",partial_paths[iter-1],data_node);



    unsigned int q_edges = 0;
    unsigned int mapped_edges = 0;

    // for each query node neighbor matching order [iter] has

    unsigned int query_node = e_p.matching_order[iter];

    unsigned int start = q_p.outgoing_neighbors_offset[query_node];
    unsigned int end = q_p.outgoing_neighbors_offset[query_node+1];
    unsigned int data_node_start = d_p.outgoing_neighbors_offset[data_node];
    unsigned int data_node_end = d_p.outgoing_neighbors_offset[data_node+1];

    //printf("\n%d out working %d %d %d %d \n",data_node, start,end,data_node_start,data_node_end);

    for(int i = start; i < end; i++ ){

        unsigned int q_n = q_p.outgoing_neighbors[i];
        // if its in the partial path
        //printf("\n check if found q_n %d, data_node %d, partialpaths %d",q_n,data_node,partial_paths[0]);
        int p_index = check_if_found(e_p.matching_order, 0, iter, q_n);
        if(p_index != -1){
            q_edges++;

            // and if the data node has that node as a neighbor
            if(check_if_found(d_p.outgoing_neighbors,data_node_start,data_node_end,partial_paths[p_index]) != -1){
                // has an edge
                mapped_edges++;
            }
            
        }
    }

    //now again for incoming paths

    start = q_p.incoming_neighbors_offset[query_node];
    end = q_p.incoming_neighbors_offset[query_node+1];
    data_node_start = d_p.incoming_neighbors_offset[data_node];
    data_node_end = d_p.incoming_neighbors_offset[data_node+1];

    //printf("\n%d inc working %d %d %d %d \n",data_node, start,end,data_node_start,data_node_end);

    for(int i = start; i < end; i++ ){

        unsigned int q_n = q_p.incoming_neighbors[i];
        // if its in the partial path
        int p_index = check_if_found(e_p.matching_order, 0, iter, q_n);
        if(p_index != -1){
            q_edges++;
            
            // and if the data node has that node as a neighbor
            if(check_if_found(d_p.incoming_neighbors,data_node_start,data_node_end,partial_paths[p_index]) != -1){
                // has an edge
                mapped_edges++;
            }
            
        }
    }

    //printf("\n Mapped: %d \t Q: %d \n",mapped_edges,q_edges);
    float score = 0.0;
    //score = mapped_edges /  q_edges 
    if(q_edges != 0){
        score = ((float)(mapped_edges)) / q_edges;
    }


    // DEBUG
    //printf("\n CANDIDATE_SCORE RESULTS - COMPARED TO %d: iter: %d, data_node: %d, score: %f \n",partial_paths[iter-1],iter,data_node,score);

    return score;



}

__device__ void find_new_good_candidates(unsigned int *partial_paths, unsigned int lane_id, unsigned int iter, unsigned int *candidates,
                                         unsigned int *buffer, unsigned int *candidate_count, G_pointers q_p, G_pointers d_p, E_pointers e_p)
{

    // Grab the Attribute of the query node we are working with
    unsigned int query_attribute = q_p.attributes[e_p.matching_order[iter]];

    // starting each thread out at their threadIdx.x
    unsigned int added_id = lane_id;
    
    while(true){

        // PICKING THREAD JOB
        unsigned int sum = 0;
        unsigned int node;
        unsigned int neighbor;
        unsigned int num_neighbors;
        bool working = false;
        // we need to find what neighbor we are looking at

        // FOR EACH MAPPED QUERY NODE ( NODES IN PARTIAL PATH )
        //      Grab the number of neighbors that node has
        //          if added id <= previous sum + ^^^^^^^
        //              Then we found our working node
        //          else
        //              add the sum together and go to next partial path node
        // 
        // added_id = 0 - 31, after each iteration, += 32
        for(int i = 0; i < iter; i++){
            num_neighbors = d_p.outgoing_neighbors_offset[partial_paths[i]+1] - d_p.outgoing_neighbors_offset[partial_paths[i]];

            //printf("\nnum_neighbors: %d, added_id:%d\n",num_neighbors,added_id);
            if(added_id < sum+num_neighbors){
                node = partial_paths[i];
                neighbor = (sum+num_neighbors)-added_id-1;
                working = true;
                break;
            } else {
                sum += num_neighbors;
            }
        }

        // if this thread doesnt have a neighbor to check, break out.
        if (!working)
        {
            break;
        }

        unsigned int data_node_checking = d_p.outgoing_neighbors[d_p.outgoing_neighbors_offset[node]+neighbor];

        //DEBUG


        // NOTE TO SELF - NEED TO MAKE SURE DATA_NODE IS NOT ALREADY MAPPED IN PARTIAL PATH


        // only check if it has the same attribute as the query
        if(d_p.attributes[data_node_checking] == query_attribute){

            if(check_if_found(partial_paths, 0, iter, data_node_checking)==-1 && candidate_score(partial_paths,iter,data_node_checking,q_p,d_p,e_p) >.49){

                // found a candidate
                unsigned int idx = atomicAdd(&candidate_count[0], 1);
                //printf("%d\n",candidate_count[0]);
                if (idx < MAX_EDGES)
                {
                    // add to canidate array
                    candidates[idx] = data_node_checking;
                }
                else
                {
                    // add to helper buffer instead
                    //printf("TIME TO PUT IN THE BUFFER \n");
                    buffer[idx - MAX_EDGES] =data_node_checking;
                }
            }    
        }
        // number of threads per warp
        added_id += 32;

    }
    




}


__device__ unsigned int get_graph_score(unsigned int *partial_paths, unsigned int iter, unsigned int data_node,
                                        unsigned int prev_score, G_pointers q_p, G_pointers d_p, E_pointers e_p)
{

    // score is saved as the following !!!@@##, where !!! are the edges in the graph, @@ are the nodes in the graph, and ## are the extra nodes in the graph

    // first, make sure this isn't a missing node
    if(check_if_found(partial_paths,0,iter,data_node)!=-1){
        // already added, missing node, return prev score
        return prev_score;
    }

    //SAME CODE FROM CANIDIDATE SCORE. WE SHOULD MAYBE FIND A WAY TO SAVE THIS BUT CRUNCHED ON TIME

    unsigned int q_edges = 0;
    unsigned int mapped_edges = 0;

    // for each query node neighbor matching order [iter] has

    unsigned int query_node = e_p.matching_order[iter];

    unsigned int start = q_p.outgoing_neighbors_offset[query_node];
    unsigned int end = q_p.outgoing_neighbors_offset[query_node+1];
    unsigned int data_node_start = d_p.outgoing_neighbors_offset[data_node];
    unsigned int data_node_end = d_p.outgoing_neighbors_offset[data_node+1];

    //printf("\n%d out working %d %d %d %d \n",data_node, start,end,data_node_start,data_node_end);

    for(int i = start; i < end; i++ ){

        unsigned int q_n = q_p.outgoing_neighbors[i];
        // if its in the partial path
        //printf("\n check if found q_n %d, data_node %d, partialpaths %d",q_n,data_node,partial_paths[0]);
        int p_index = check_if_found(e_p.matching_order, 0, iter, q_n);
        if(p_index != -1){
            q_edges++;

            // and if the data node has that node as a neighbor
            if(check_if_found(d_p.outgoing_neighbors,data_node_start,data_node_end,partial_paths[p_index]) != -1){
                // has an edge
                mapped_edges++;
            }
            
        }
    }

    //now again for incoming paths

    start = q_p.incoming_neighbors_offset[query_node];
    end = q_p.incoming_neighbors_offset[query_node+1];
    data_node_start = d_p.incoming_neighbors_offset[data_node];
    data_node_end = d_p.incoming_neighbors_offset[data_node+1];

    //printf("\n%d inc working %d %d %d %d \n",data_node, start,end,data_node_start,data_node_end);

    for(int i = start; i < end; i++ ){

        unsigned int q_n = q_p.incoming_neighbors[i];
        // if its in the partial path
        int p_index = check_if_found(e_p.matching_order, 0, iter, q_n);
        if(p_index != -1){
            q_edges++;
            
            // and if the data node has that node as a neighbor
            if(check_if_found(d_p.incoming_neighbors,data_node_start,data_node_end,partial_paths[p_index]) != -1){
                // has an edge
                mapped_edges++;
            }
            
        }
    }
    unsigned int score = prev_score + 10000*mapped_edges + 100*1;


    return score;
}

__device__ void print_precopy_plus_node(unsigned int* partial_path, unsigned int iter, unsigned int node, unsigned int score){

    printf("\n[%d,%d,%d,%d,%d,%d,%d,%d]\n",partial_path[0],partial_path[1],partial_path[2],partial_path[3],partial_path[4],partial_path[5],node,score);

}

__device__ unsigned int get_score_specific(unsigned int mode, unsigned int score){
    if(mode == 0){
        // return number of edges
        return score/10000;
    } else if (mode == 1){
        // return number of nodes
        return ((score/100)%100);
    } else {
        // return intra nodes
        return (score % 100);
    }
    return 0;
}

__device__ bool is_core_node(unsigned int node, G_pointers q_p){
    if (q_p.signatures[node*Signature_Properties+0] >= 2 || q_p.signatures[node*Signature_Properties+1] >= 2 ){
        return true;
    } else {
        return false;
    }
}

__global__ void approximate_search_kernel(G_pointers query_pointers, G_pointers data_pointers, E_pointers extra_pointers,
                                          unsigned int iter, unsigned int jobs)
{

    // this is one of the lists used in the cuTS implementation.
    __shared__ unsigned int partial_paths[WARPS_EACH_BLK * MAX_QUERY_NODES]; // 24,576 ( 3.072 kb )

    __shared__ unsigned int cand_counter[WARPS_EACH_BLK]; // 48

    __shared__ unsigned int temp_neighbors[WARPS_EACH_BLK * MAX_EDGES]; // 229,376 ( 28.672 kb )
    __shared__ unsigned int candidates[WARPS_EACH_BLK * MAX_EDGES];      // 229,376 ( 28.672 kb )

    // total shared mem used = 60.422 kilobytes per block

    unsigned int v = extra_pointers.matching_order[iter];
    unsigned int mask = 0xFFFFFFFF;
    unsigned int warp_id = threadIdx.x / 32;
    unsigned int lane_id = threadIdx.x % 32;
    unsigned int global_idx = (blockIdx.x) * WARPS_EACH_BLK + warp_id;

    extra_pointers.global_count[0] = 0;
    printf("T: %d,W: %d,L: %d,G: %d\n",threadIdx.x,warp_id,lane_id,global_idx);

    
    __syncthreads();

    while (true)
    {
        __syncwarp();

        unsigned int pre_idx;

        // first thread in each warp (  threadIdx.x%32; )
        if (lane_id == 0)
        {
            pre_idx = atomicAdd(&extra_pointers.global_count[0], 1);
        }
        __syncthreads();
        if(lane_id == 0 && iter == 3){
            //printf("%d,",pre_idx);
        }

            

        pre_idx = __shfl_sync(mask, pre_idx, 0);

        if (pre_idx >= jobs)
        {
            if(iter == 1 && lane_id == 0){
                //printf("%d warp is out\n",global_idx);
            }
            
            break;
        }

        
        // construct partial paths
        construct_partial_path(partial_paths, cand_counter, extra_pointers.result_lengths, extra_pointers.indexes_table, extra_pointers.results_table,
                               lane_id, warp_id, iter, pre_idx);

        if(lane_id==0 && iter == 1){
            //printf("%d\n",pre_idx);
            //printf("\nPARTIAL PATH: %d \n",partial_paths[warp_id*MAX_QUERY_NODES+iter-1]);
        }
 


        find_new_good_candidates(&partial_paths[warp_id * MAX_QUERY_NODES], lane_id, iter, &candidates[warp_id * MAX_EDGES],
                            &extra_pointers.helper_buffer[global_idx * BUFFER_PER_WARP], &cand_counter[warp_id],
                            query_pointers, data_pointers, extra_pointers);

        if(lane_id==0){
            //printf("\npre idx: %d ---- CANDS: %d\n",pre_idx,cand_counter[warp_id]);
        }

        // we need to sort out duplicate candidates
        sort_out_dupes(&candidates[warp_id * MAX_EDGES], &extra_pointers.helper_buffer[global_idx * BUFFER_PER_WARP], &cand_counter[warp_id], lane_id);

        if(lane_id==0 && cand_counter[warp_id] > 0 && iter==2){
            //printf("jobs: %d, pre idx: %d, global count: %d WLG:%d,%d,%d---- CANDS AFTER SORT: %d\n",jobs,pre_idx,extra_pointers.global_count[0],warp_id,lane_id,global_idx,cand_counter[warp_id]);
            //printf("%d,",cand_counter[warp_id]);
        }

        if(cand_counter[warp_id] <= 0){
            continue;
        }
        // now, we can write out the values
        if(iter == query_pointers.V -1){
            unsigned int write_offset;

            if(lane_id==0){
                //for(int i = 0; i<cand_counter[warp_id]; i++){
                    //print_precopy_plus_node(&partial_paths[warp_id * MAX_QUERY_NODES],iter,candidates[warp_id * MAX_EDGES+i]);
                //}
                atomicAdd(&extra_pointers.write_pos[0], cand_counter[warp_id]);
            }


        } else {

        // save this for knowing where we're writing too
        unsigned int write_offset;
        if(lane_id==0){
            //printf("\nFOUND CAND - %d",candidates[0]);
        }
        

            // for each found candidate
        for (unsigned int i = lane_id; i < cand_counter[warp_id]; i += 32)
        {
            unsigned int val = get_val_from_array_and_buf(&candidates[warp_id * MAX_EDGES], &extra_pointers.helper_buffer[global_idx * BUFFER_PER_WARP], MAX_EDGES, i);
                // increase the offset and save them in the results and index table
            write_offset = atomicAdd(&extra_pointers.result_lengths[iter+1], 1);
            //printf("%d\n",write_offset);
            //printf("\nwrite offset: %d",write_offset);
            extra_pointers.results_table[write_offset] = val;
            extra_pointers.indexes_table[write_offset] = pre_idx + extra_pointers.result_lengths[iter - 1];
            // unused atm
            // extra_pointers.intra_v_table
            unsigned int prev_score = extra_pointers.scores_table[pre_idx+extra_pointers.result_lengths[iter - 1]];
            extra_pointers.scores_table[write_offset] = get_graph_score(&partial_paths[warp_id * MAX_QUERY_NODES], iter, val, prev_score,
                                                                        query_pointers,data_pointers,extra_pointers);
        }

        // if there isn't already a missing node in this graph, and if this iter is not a core node
        unsigned int prev_score = extra_pointers.scores_table[pre_idx+extra_pointers.result_lengths[iter - 1]];
        if(get_score_specific(1,prev_score) >= iter && !is_core_node(v,query_pointers)){

            //make it a missing node
            write_offset = atomicAdd(&extra_pointers.result_lengths[iter+1], 1);
            // write an empty node
            unsigned int val = extra_pointers.results_table[pre_idx+extra_pointers.result_lengths[iter - 1]];

            extra_pointers.results_table[write_offset] = val;
            extra_pointers.indexes_table[write_offset] = pre_idx + extra_pointers.result_lengths[iter - 1];
            extra_pointers.scores_table[write_offset] = get_graph_score(&partial_paths[warp_id * MAX_QUERY_NODES], iter, val, prev_score,
                                                                        query_pointers,data_pointers,extra_pointers);

        }
    }
}
