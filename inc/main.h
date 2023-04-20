#ifndef MAIN_H
#define MAIN_H

#include "common.h"
#include "graph.h"
#include "memory.h"
#include "device.h"


unsigned long long int approx_searching(string query_file,string data_file,unsigned int k_top);

void print_graph(unsigned int * indexes, unsigned int * result, unsigned int start_pos);

unsigned int get_score_specific_cpu(unsigned int mode, unsigned int score);

void iterate_over_results(unsigned int k_top, unsigned int * results, unsigned int * indexes, unsigned int * scores, unsigned int *  size, Graph q);

#endif