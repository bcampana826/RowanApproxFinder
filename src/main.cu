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



    //generate_matching_order<<<1,1>>>(query_pointers,extra_pointers.matching_order,data_pointers.attributes_in_order_offset);

    std::cout << "matching = ";
    for (unsigned int i = 0; i < query_graph.V; i++) {
        std::cout << query_graph.matching_order[i] << " ";
    }
    



    /**
     * cout<<"Query..."<<endl;
     * query_graph.printGraph();
     *  cout<<"Data..."<<endl;
     *  data_graph.printGraph();
    */

    // Find Root and Matching Order
    cout<<"Testing Node Viability Score"<<endl;
    //testing<<<1,1>>>(query_pointers,data_pointers,extra_pointers);





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

/**
 *

void gen_root_node_and_matching_order(Graph query, Graph data){

    query.order_sequence = new unsigned int[query.V];

    // find the root node first
    for(unsigned int i=0; i<query.V; i++){

        // if this node only has less than two neighbors, do not include
        //      hmm, for this, im only going to consider outgoing for now
        //      definetly worth taking a re-examination of this
        if(query.signatures[i*Signature_Properties]+Out_degree_offset < 2){
            continue;
        }

        


    }

*/

