#ifndef COMMON_H
#define COMMON_H

#include <string>
#include <vector>
#include <set>
#include <fstream>
#include <iostream>
#include "omp.h"

#define Signature_Properties 2
#define In_degree_offset 0
#define Out_degree_offset 1

using namespace std;

typedef struct G_pointers {
    unsigned int* outgoing_neighbors;
    unsigned int* outgoing_neighbors_offset;
    unsigned int* signatures;
    unsigned int* incoming_neighbors; 
    unsigned int* incoming_neighbors_offset;
    unsigned int* attributes;
    unsigned int* attributes_in_order;
    unsigned int* attributes_in_order_offset;
    unsigned int V;
    unsigned int largest_att;
    unsigned int* matching_order;
} G_pointers; //graph related

#endif