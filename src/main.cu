#include "../inc/main.h"

class QueueValue
{
   unsigned int node;
   float score;
public:
   QueueValue(unsigned int node, float score)
   {
      this->node = node;
      this->score = score;
   }
   unsigned int getNode() const { return node; }
   float getScore() const { return score; }
};

class QueueComparator
{
public:
    int operator() (const QueueValue& q1, const QueueValue& q2)
    {
        return q1.getScore() > q2.getScore();
    }
};

unsigned long long int approx_searching(string query_file,string data_file,unsigned int k_top){


    // Read in the Query File and the Data File
    cout<<"start loading graph file from disk to memory..."<<endl;
    Graph query_graph(true,query_file);

    cout<<"Query Loaded!"<<endl;

    Graph data_graph(false,data_file);

    query_graph.create_matching_order(data_graph);

    G_pointers query_pointers;
    G_pointers data_pointers;

    E_pointers extra_pointers;


    // Save them both over to the GLOBAL memory
    cout<<"start copying graph to gpu..."<<endl;
    malloc_graph_to_gpu_memory(query_graph,query_pointers,true);
    malloc_graph_to_gpu_memory(data_graph,data_pointers,false);

    malloc_extra_to_gpu_memory(extra_pointers,query_graph.V,query_graph.matching_order);
    cout<<"end copying graph to gpu..."<<endl;


    // Init Searching on that Root
    initialize_searching<<<BLK_NUMS,BLK_DIM>>>(query_pointers,data_pointers,extra_pointers);
    cudaDeviceSynchronize();

    cout<<"Query Nodes: "<<query_graph.V<<endl;

    cout<<"Initial Canidates: "<<extra_pointers.result_lengths[1]<<endl;


    // if we have at least one match
    if(extra_pointers.result_lengths[1] > 0){
        
        for(unsigned int iter = 1; iter < query_graph.V; iter++){

            extra_pointers.result_lengths[iter+1] = extra_pointers.result_lengths[iter];

            unsigned int jobs_count = extra_pointers.result_lengths[iter] - extra_pointers.result_lengths[iter-1];

            
            cout<<"\niter: "<<iter<<" Jobs: "<<jobs_count<<endl;
            cout<<"Finding "<<query_graph.matching_order[iter]<<endl;
    

            approximate_search_kernel<<<BLK_NUMS,BLK_DIM>>>(query_pointers,data_pointers,extra_pointers,iter,jobs_count);
            cudaDeviceSynchronize();

            cout<<"Results: "<<extra_pointers.result_lengths[iter+1]<<endl;



            cout<<"Results: "<<extra_pointers.result_lengths[iter+1]<<endl;

          // Allocate CPU memory
        unsigned int* results = (unsigned int*)malloc(extra_pointers.result_lengths[query_graph.V]);  // Allocate memory on the CPU for results_table
        unsigned int* indexes = (unsigned int*)malloc(extra_pointers.result_lengths[query_graph.V]);  // Allocate memory on the CPU for indexes_table
        unsigned int* scores = (unsigned int*)malloc(extra_pointers.result_lengths[query_graph.V]);   // Allocate memory on the CPU for scores_table

        cudaMemcpy(results, extra_pointers.results_table, sizeof(unsigned int) * extra_pointers.result_lengths[query_graph.V], cudaMemcpyDeviceToHost);
        cudaMemcpy(indexes, extra_pointers.indexes_table, sizeof(unsigned int) * extra_pointers.result_lengths[query_graph.V], cudaMemcpyDeviceToHost);
        cudaMemcpy(scores, extra_pointers.results_table, sizeof(unsigned int) * extra_pointers.result_lengths[query_graph.V], cudaMemcpyDeviceToHost);


        }


        
        printf("\n\n\n%d",extra_pointers.result_lengths[query_graph.V]);
        iterate_over_results(k_top, results, indexes, scores, &extra_pointers.result_lengths[query_graph.V], query_graph);
    }


    return 0;
    /**
     * First, it will find the root node, this can probably be parallel but its
     * only on the query file so it wont be that much work anyway. This will also
     * create the matching order
     * 
     * This method can be found in seed_finder.h of GFinder
     * 
    */

    // Init Searching on that Root

    /**
     * This will be the first parallel method.
     * 
     * Essentially, Its going to find all of the initial canidates of the first node in 
     * the matching order, and save them to the INS table
     * 
     * The rest will be done in a looped method, like in CUTS
     * 
    */

   // Search Loop

   /**
    * This is the real meat and potatoes
    * 
    * Just like cuts, most of the work will be done in this method
    * 
    * Once this method is run it will not stop till its done with all verticies
    * 
   */

   // Iterate over INS table and pick the K highest scores, then print / write to file


}

void print_graph(unsigned int * indexes, unsigned int * result, unsigned int start_pos){
    
    unsigned int idx = start_pos;
    
    unsigned int * list = new unsigned int[MAX_QUERY_NODES];
    unsigned int size = 0;


    while (idx > 0)
    {
        /* code */
        list[size] = result[idx];
        size++;
        idx = indexes[idx];
    }
    
    printf("(");
    for(int i = size; i >= 0; i--){
        printf("%d,",list[i]);
    }
    printf(")\n");

}

unsigned int get_score_specific_cpu(unsigned int mode, unsigned int score){
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


void iterate_over_results(unsigned int k_top, unsigned int * results, unsigned int * indexes, unsigned int * scores, unsigned int *  size, Graph q){
    // Create a max heap using priority_queue with custom comparator
    printf("min heap");
    std::priority_queue<QueueValue, vector<QueueValue>, QueueComparator> min_heap;


    // Insert elements from the list into the max heap
    for (unsigned int i = size[0]; i>0; i--) {
        float e_count = q.E - get_score_specific_cpu(0,scores[i]);
        float v_count = q.V - get_score_specific_cpu(1,scores[i]);
        float iv_count = get_score_specific_cpu(2,scores[i]);

        float temp = q.E * WEIGHT_MISSING_EDGE + q.V * WEIGHT_MISSING_VERT + WEIGHT_INTRA_VERT * iv_count;

        printf("%f,",temp);

        min_heap.push(QueueValue(indexes[i],temp));
    }

    printf("top n");
    std::vector<QueueValue> top_N_values;
    for (int i = 0; i < k_top; ++i) {
        top_N_values.push_back(min_heap.top());
        min_heap.pop();
    }

    printf("print_graph");
    for(int i = 0; i<k_top; i++){
        print_graph(indexes, results, top_N_values[i].getNode());      
    }

    
}


