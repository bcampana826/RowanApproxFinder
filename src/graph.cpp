#include "../inc/graph.h"
#include "../inc/util.h"

Graph::Graph(bool mode, string input_file){

    std::cout << "Creating Graph..." << std::endl;

    vector<set<unsigned int>> outgoing_neighbors_set;
    vector<set<unsigned int>> incoming_neighbors_set;
    vector<set<unsigned int>> attribute_set;

    /**
     * After this method, V, E, are saved, attributes array is created and filled
     * attribute set is not filled in yet, this will be filled in on next for loop
    */
    read_graph_file(input_file, outgoing_neighbors_set, incoming_neighbors_set, attribute_set);

    signatures = new unsigned int[V*Signature_Properties];

    #pragma omp parallel for
    for(int i=0;i<V;++i){
        signatures[i*Signature_Properties+In_degree_offset] = incoming_neighbors_set[i].size();
        signatures[i*Signature_Properties+Out_degree_offset] = outgoing_neighbors_set[i].size();
    }

    incoming_neighbors_offset = new unsigned int[V+1];
    outgoing_neighbors_offset = new unsigned int[V+1];


    #pragma omp parallel for
    for(unsigned int i=0;i<V+1;++i){
        incoming_neighbors_offset[i] = 0;
        outgoing_neighbors_offset[i] = 0;
    }

    // pre load the attributes
    attributes_in_order_offset = new unsigned int[largest_att+2];

    std::cout << "Before save..." << std::endl;

    #pragma omp parallel for
    for(unsigned int i=0;i<largest_att+2;++i){
        attributes_in_order_offset[i] = 0;
    }

    

    unsigned int bound = (V > largest_att) ? V : largest_att;
    
    std::cout << "Before fill..." << std::endl;

    for(unsigned int i=1;i<bound+2;++i){

        if(i < V+1){
            incoming_neighbors_offset[i] += incoming_neighbors_offset[i-1] + incoming_neighbors_set[i-1].size();
            outgoing_neighbors_offset[i] += outgoing_neighbors_offset[i-1] + outgoing_neighbors_set[i-1].size();
        }
        if(i < largest_att+2){
            // this is +2. this is because attributes might not be 0 based
            attributes_in_order_offset[i] += attributes_in_order_offset[i-1] + attribute_set[i-1].size();
        }
    }


    std::cout << "Before iterate..." << std::endl;

    outgoing_neighbors = new unsigned int[outgoing_neighbors_offset[V]];
    incoming_neighbors = new unsigned int[incoming_neighbors_offset[V]];
    attributes_in_order = new unsigned int[attributes_in_order_offset[largest_att]];


     std::cout << largest_att << std::endl;

    unsigned int j = 0;
    unsigned int k = 0;
    unsigned int l = 0;
    for(unsigned int i=0;i<bound+1;++i){
        set<unsigned int> s;

        if(i < V){
            s = outgoing_neighbors_set[i];
            for(std::set<unsigned int>::iterator p = s.begin();p!=s.end();p++){
                outgoing_neighbors[j] = *p;
                j++;
            }
            s = incoming_neighbors_set[i];
            for(std::set<unsigned int>::iterator p = s.begin();p!=s.end();p++){
                incoming_neighbors[k] = *p;
                k++;
            }
        }

        if(i < largest_att+1){

            std::cout << i << "before" << std::endl;

            s = attribute_set[i];
            
            for(std::set<unsigned int>::iterator p = s.begin();p!=s.end();p++){
                
                attributes_in_order[l] = *p;
                std::cout << attributes_in_order[l] << std::endl;
                l++;
            }

        }

    }

    std::cout << "done..." << std::endl;

}


void Graph::read_graph_file(string input_file, vector<set<unsigned int>> &outgoing_neighbors_set,vector<set<unsigned int>> &incoming_neighbors_set, vector<set<unsigned int>> &attribute_set){

    std::cout << "Reading Graph..."<< std::endl;

    double load_start = omp_get_wtime();
    ifstream infile;
    infile.open(input_file);
    if(!infile){
        cout<<"load graph file failed "<<endl;
        exit(-1);
    }

    string line;
    const string delimter = " ";
    unsigned int line_index = 0;
    
    vector<string> split_list;

    // Grab the first line in the file and save the number of vertexes
    getline(infile, line);
    split(line, split_list, ' ');
    V = stoi(split_list[2]);


    for(unsigned int i=0;i<V;++i){
        set<unsigned int> temp1;
        set<unsigned int> temp2;
        outgoing_neighbors_set.push_back(temp1);
        incoming_neighbors_set.push_back(temp2);
    }


    attributes = new unsigned int[V];

    unsigned int counter = 0;
    E = 0;

    set<unsigned int> unique_attributes;
    largest_att = 0;

    while (getline(infile, line)) {
        
        if (line.at(0) == 'v'){
            split(line, split_list, ' ');

            attributes[counter] = stoi(split_list[2]);
            unique_attributes.insert(stoi(split_list[2]));
            counter++;

            if (stoi(split_list[2]) > largest_att){
                largest_att = stoi(split_list[2]);
            }


        }
        else if (line.at(0) == 'e'){
            split(line, split_list, ' ');

            int left = stoi(split_list[1]);
            int right = stoi(split_list[2]);

            outgoing_neighbors_set[left].insert(right);
            incoming_neighbors_set[right].insert(left);

            E++;
        }
    }
    std::cout << largest_att<< std::endl;
    std::cout << "before atts..."<< std::endl;


    // currently, this makes sure we have a list for all potential attributes
    // some may be unused, like if our only attributes are 100 and 150, then
    // we have 148 unused lists, definately something to look at
    for(int i = 0; i < largest_att+1; i++){
        set<unsigned int> temp1;
        attribute_set.push_back(temp1);
    }

    std::cout << "after first atts..."<< std::endl;

    for(int i = 0; i < V; i++){
        attribute_set[attributes[i]].insert(i);
    }

    infile.close();
    double load_end = omp_get_wtime();

    std::cout << "Reading Graph Complete..."<< std::endl;

}



void Graph::printGraph() {

        std::cout << "Printing Graph..."<< std::endl;

        std::cout << "V = " << V << std::endl;
        std::cout << "E = " << E << std::endl;
        std::cout << "Largest Atts = " << largest_att << std::endl;
   //     std::cout << "AVG_DEGREE = " << AVG_DEGREE << std::endl;

        std::cout << "attributes = ";
        for (unsigned int i = 0; i < V; i++) {
            std::cout << attributes[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "attributes_in_order_offset = ";
        for (unsigned int i = 0; i < largest_att+2; i++) {
            std::cout << attributes_in_order_offset[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "attributes_in_order = ";
        for (unsigned int i = 0; i < V ; i++) {
            std::cout << attributes_in_order[i] << " ";
        }
        std::cout << std::endl;


        std::cout << "incoming_neighbors = ";
        for (unsigned int i = 0; i < E; i++) {
            std::cout << incoming_neighbors[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "outgoing_neighbors = ";
        for (unsigned int i = 0; i < E; i++) {
            std::cout << outgoing_neighbors[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "incoming_neighbors_offset = ";
        for (unsigned int i = 0; i < V+1; i++) {
            std::cout << incoming_neighbors_offset[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "outgoing_neighbors_offset = ";
        for (unsigned int i = 0; i < V+1; i++) {
            std::cout << outgoing_neighbors_offset[i] << " ";
        }
        std::cout << std::endl;

/**
        std::cout << "parents_offset = ";
        for (unsigned int i = 0; i < V; i++) {
            std::cout << parents_offset[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "parents = ";
        for (unsigned int i = 0; i < E; i++) {
            std::cout << parents[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "children = ";
        for (unsigned int i = 0; i < E; i++) {
            std::cout << children[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "children_offset = ";
        for (unsigned int i = 0; i < V; i++) {
            std::cout << children_offset[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "order_sequence = ";
        for (unsigned int i = 0; i < V; i++) {
            std::cout << order_sequence[i] << " ";
        }
        std::cout << std::endl;
*/
        std::cout << "signatures = ";
        for (unsigned int i = 0; i < Signature_Properties*V; i++) {
            std::cout << signatures[i] << " ";
        }
        std::cout << std::endl;


}