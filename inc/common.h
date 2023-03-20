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
    unsigned int* attributes;
    unsigned int* incoming_neighbors; 
    unsigned int* incoming_neighbors_offset;
    unsigned int V;
} G_pointers; //graph related
