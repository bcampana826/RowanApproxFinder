
#ifndef GRAPH_H
#define GRAPH_H

#include "./common.h"

class Graph{
public:
    unsigned int V;
    unsigned int E;
    unsigned int num_attributes;
    unsigned int Root;
    unsigned int AVG_DEGREE = 0;

    unsigned int * attributes;

    unsigned int * attributes_in_order;
    unsigned int * attributes_in_order_offset;
    
    unsigned int * incoming_neighbors;
    unsigned int * outgoing_neighbors;

    unsigned int * incoming_neighbors_offset;
    unsigned int * outgoing_neighbors_offset;

    unsigned int * matching_order;
    unsigned int * signatures;
    
    void read_graph_file(string input_file, vector<set<unsigned int>> &outgoing_neighbors_set,vector<set<unsigned int>> &incoming_neighbors_set, vector<set<unsigned int>> &attribute_set);
    void gen_root_node_and_matching_order();
    void printGraph();
    void create_matching_order(Graph data);
    Graph(bool mode,std::string input_file);
    //mode 0 for query graph, mode 1 for data graph
};

float get_node_viability_score(unsigned int query_node, unsigned int *query_atts, unsigned int *data_atts_orders_offset,
                                          unsigned int *query_signatures);


#endif