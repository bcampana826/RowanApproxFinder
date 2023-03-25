#ifndef MEMORY_H
#define MEMORY_H

#include "./graph.h"
#include "./common.h"

void malloc_graph_to_gpu_memory(Graph &g, G_pointers &p, bool query);

#endif