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

    // empty spot for leading offset
    attributes_in_order_offset = new unsigned int[(num_attributes+1)];

    std::cout << "Before save..." << std::endl;

    #pragma omp parallel for
    for(unsigned int i=0, end = num_attributes + 1; i < end;i++){
        attributes_in_order_offset[i] = 0;
    }

    
    std::cout << "Before fill..." << std::endl;

    for(unsigned int i=1, end = 1 + ((V > num_attributes) ? (V+1) : num_attributes+1);
        i < end;++i){

        if(i < V+1){
            incoming_neighbors_offset[i] += incoming_neighbors_offset[i-1] + incoming_neighbors_set[i-1].size();
            outgoing_neighbors_offset[i] += outgoing_neighbors_offset[i-1] + outgoing_neighbors_set[i-1].size();
        }
        if(i < (num_attributes+1)){
            attributes_in_order_offset[i] += attributes_in_order_offset[i-1] + attribute_set[i-1].size();
        }
    }


    std::cout << "Before iterate..." << std::endl;

    outgoing_neighbors = new unsigned int[outgoing_neighbors_offset[V]];
    incoming_neighbors = new unsigned int[incoming_neighbors_offset[V]];
    // should not this be num_attributes ?
    attributes_in_order = new unsigned int[attributes_in_order_offset[num_attributes]];


    std::cout << num_attributes << std::endl;

    unsigned int j = 0;
    unsigned int k = 0;
    unsigned int l = 0;

    for(unsigned int i=0; i<V; ++i){
        set<unsigned int> s;
        s = outgoing_neighbors_set[i];
        for(set<unsigned int>::iterator p = s.begin();p!=s.end();p++){
            outgoing_neighbors[j] = *p;
            j++;
        }
        s = incoming_neighbors_set[i];
        for(set<unsigned int>::iterator p = s.begin();p!=s.end();p++){
            incoming_neighbors[k] = *p;
            k++;
        }

    }

    std::cout << "Before iterate..." << std::endl;

    for(unsigned int i=0;i<num_attributes;++i){

        set<unsigned int> s;
        //std::cout << "looping atts " << i << std::endl;
        s = attribute_set[i];
        //std::cout << s.size() << std::endl;
        for(set<unsigned int>::iterator p = s.begin();p!=s.end();p++){

            attributes_in_order[l] = *p;
            //std::cout << l << std::endl;
            l++;
        }
    }

    //std::cout << "Gen 2 Hop.." << endl;
    //generate_2_hop_arrays();


    std::cout << "Done..." << std::endl;
}

void Graph::generate_2_hop_arrays(){
    
    vector<set<unsigned int>> two_hop_set;
    

    for(unsigned int i=0;i<V;i++){
        set<unsigned int> temp1;
        two_hop_set.push_back(temp1);
    }

    std::cout << "Crazy Method.." << endl;
    #pragma omp parallel for
    for(unsigned int i=0;i < V;i++){



        unsigned int two_start;
        unsigned int two_end;
        unsigned int node;
        unsigned int one_start = outgoing_neighbors_offset[i];
        unsigned int one_end = outgoing_neighbors_offset[i+1];
        
        for(unsigned int j=0;j<one_end-one_start;j++){


            two_start = outgoing_neighbors_offset[outgoing_neighbors[one_start+j]];
            two_end = outgoing_neighbors_offset[outgoing_neighbors[one_start+j]+1];
            for(unsigned int k = 0; k<two_end-two_start; k++){
                node = outgoing_neighbors[two_start+k];
                if(node != i){
                    two_hop_set[i].insert(node);
                }
            }


            two_start = incoming_neighbors_offset[outgoing_neighbors[one_start+j]];
            two_end = incoming_neighbors_offset[outgoing_neighbors[one_start+j]+1];
            for(unsigned int k = 0; k<two_end-two_start; k++){
                node = incoming_neighbors[two_start+k];
                if(node != i){
                    two_hop_set[i].insert(node);
                }
            }
        }

        one_start = incoming_neighbors_offset[i];
        one_end = incoming_neighbors_offset[i+1];
        for(unsigned int j=0;j<one_end-one_start;j++){

            two_start = outgoing_neighbors_offset[incoming_neighbors[one_start+j]];
            two_end = outgoing_neighbors_offset[incoming_neighbors[one_start+j]+1];
            for(unsigned int k = 0; k<two_end-two_start; k++){
                node = outgoing_neighbors[two_start+k];
                if(node != i){
                    two_hop_set[i].insert(node);
                }
            }


            two_start = incoming_neighbors_offset[incoming_neighbors[one_start+j]];
            two_end = incoming_neighbors_offset[incoming_neighbors[one_start+j]+1];
            for(unsigned int k = 0; k<two_end-two_start; k++){
                node = incoming_neighbors[two_start+k];
                if(node != i){
                    two_hop_set[i].insert(node);
                }
            }
        }

    }

    std::cout << "after.." << endl;

    two_hop_neighbors_offset = new unsigned int[V+1];
    #pragma omp parallel for
    for(unsigned int i = 0; i < V+1; i++){
        two_hop_neighbors_offset[i] = 0;
    }


    for(unsigned int i = 1; i < V+1; i++){
        two_hop_neighbors_offset[i] += two_hop_neighbors_offset[i-1] + two_hop_set[i-1].size();
    }

    two_hop_neighbors = new unsigned int[two_hop_neighbors_offset[V]];

    unsigned int k=0;
    for(unsigned int i=0; i<V; ++i){
        set<unsigned int> s;
        s = two_hop_set[i];
        for(set<unsigned int>::iterator p = s.begin();p!=s.end();p++){
            two_hop_neighbors[k] = *p;
            k++;
        }
    }
}

void Graph::read_graph_file(string input_file, vector<set<unsigned int>> &outgoing_neighbors_set,vector<set<unsigned int>> &incoming_neighbors_set, vector<set<unsigned int>> &attribute_set){

    std::cout << "Reading Graph..."<< std::endl;

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

   // set<unsigned int> unique_attributes;
    // if we end up using set unique_attr
    // def remove finding num_attributes manually (below)
    num_attributes = 0;

    while (getline(infile, line)) {
        
        if (line.at(0) == 'v'){
            split(line, split_list, ' ');

            attributes[counter] = stoi(split_list[2]);
       //     unique_attributes.insert(stoi(split_list[2]));
            counter++;

            if (stoi(split_list[2]) > num_attributes) {
                num_attributes = stoi(split_list[2]);
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

    // + 1 to max label = num labels
    ++num_attributes;

    //std::cout << num_attributes<< std::endl;
    //std::cout << "before atts..."<< std::endl;


    for(int i = 0; i < num_attributes; i++){
        set<unsigned int> temp1;
        attribute_set.push_back(temp1);
    }

    //std::cout << "after first atts..."<< std::endl;

    for(int i = 0; i < V; i++){

        // grab V's attribute
        // insert V into attribute_set[at att]

        attribute_set[attributes[i]].insert(i);
    }

    infile.close();

    std::cout << "Reading Graph Complete..."<< std::endl;

}

void Graph::printGraph() {

        std::cout << "Printing Graph..."<< std::endl;

        std::cout << "V = " << V << std::endl;
        std::cout << "E = " << E << std::endl;
        std::cout << "Number of attributes = " << num_attributes << std::endl;
   //     std::cout << "AVG_DEGREE = " << AVG_DEGREE << std::endl;

        std::cout << "attributes = ";
        for (unsigned int i = 0; i < V; i++) {
            std::cout << attributes[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "attributes_in_order_offset = ";
        for (unsigned int i = 0; i < num_attributes+1; i++) {
            std::cout << attributes_in_order_offset[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "attributes_in_order = ";
        for (unsigned int i = 0; i < V ; i++) {
            std::cout << attributes_in_order[i] << " ";
        }
        std::cout << std::endl;


        std::cout << "incoming_neighbors = ";
        for (unsigned int i = 0; i < incoming_neighbors_offset[V]; i++) {
            std::cout << incoming_neighbors[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "outgoing_neighbors = ";
        for (unsigned int i = 0; i < outgoing_neighbors_offset[V]; i++) {
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

        std::cout << "signatures = ";
        for (unsigned int i = 0; i < Signature_Properties*V; i++) {
            std::cout << signatures[i] << " ";
        }
        std::cout << "Graph Printed." << std::endl;


}

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

void Graph::create_matching_order(Graph d){
    // first, go through all the nodes and find the root node
    //      needs to have at least 2 neighbors
    //      needs to have the lowest node_viability_score
    unsigned int min_node = 0;
    float min_score = 10000000.0;
    float node_viability = 0.0;

    for(int i = 0; i<V; i++){

        if (signatures[i*Signature_Properties+0] >= 2 || signatures[i*Signature_Properties+1] >= 2 ){
            // this is a core node.
            node_viability = get_node_viability_score(i, attributes, d.attributes_in_order_offset, signatures);

            if ( min_score > node_viability ){
                min_score = node_viability;
                min_node = i;
            }
        }
    }

    unsigned int count = 0;
    unsigned int node;
    unsigned int start;
    unsigned int end;
    unsigned int neighbor;

    set<unsigned int> added;
    priority_queue <QueueValue, vector<QueueValue>, QueueComparator > pq;

    pq.push(QueueValue(min_node,min_score));

    matching_order = new unsigned int[V];
    
    while(!pq.empty()){

        // grab out of the heap
        QueueValue value = pq.top();
        pq.pop();

        node = value.getNode();

        // add to the order
        matching_order[count] = node;
        count++;
        added.insert(node);

        // now iterate over this nodes outgoing children and add them to the queue
        start = outgoing_neighbors_offset[node];
        end = outgoing_neighbors_offset[node+1];

        for(int i = 0; i < end - start; i++){

            neighbor = outgoing_neighbors[start+i];           

            if(!added.count(neighbor)){
                added.insert(neighbor);
                pq.push(QueueValue(neighbor, get_node_viability_score(neighbor, attributes, d.attributes_in_order_offset, signatures)));
            }
        }

        // now iterate over this nodes incoming children and add them to the queue
        start = incoming_neighbors_offset[node];
        end = incoming_neighbors_offset[node+1];

        for(int i = 0; i < end - start; i++){

            neighbor = incoming_neighbors[start+i];           

            if(!added.count(neighbor)){
                added.insert(neighbor);
                pq.push(QueueValue(neighbor, get_node_viability_score(neighbor, attributes, d.attributes_in_order_offset, signatures)));
            }
        }

    }

    std::cout << "Matching Order = ";
    for (unsigned int i = 0; i < V; i++) {
        std::cout << matching_order[i] << " ";
    }




}

/**
 * this is the score
 * |C(node)| / degree   
 * 
 * it only considers out neighbors for degree
*/
float get_node_viability_score(unsigned int query_node, unsigned int *query_atts, unsigned int *data_atts_orders_offset,
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