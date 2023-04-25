#include "../inc/main.h"



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



        }

        cout<<"\n\n\n DONE: Final Exact Matches = "<<extra_pointers.write_pos[0]<<endl;



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


