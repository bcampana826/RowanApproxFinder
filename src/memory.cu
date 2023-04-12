#include "../inc/memory.h"

inline void chkerr(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        std::cout << cudaGetErrorString(code) << std::endl;
        exit(-1);
    }
}

void malloc_graph_to_gpu_memory(Graph &g, G_pointers &p, bool query)
{

    // std::cout<<"1"<<std::endl;
    chkerr(cudaMalloc(&(p.outgoing_neighbors), g.outgoing_neighbors_offset[g.V] * sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.outgoing_neighbors, g.outgoing_neighbors, g.outgoing_neighbors_offset[g.V] * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // std::cout<<"2"<<std::endl;
    chkerr(cudaMalloc(&(p.outgoing_neighbors_offset), (g.V + 1) * sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.outgoing_neighbors_offset, g.outgoing_neighbors_offset, (g.V + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // signatures
    // std::cout<<"3"<<std::endl;
    chkerr(cudaMalloc(&(p.signatures), (g.V) * sizeof(unsigned int) * Signature_Properties));
    chkerr(cudaMemcpy(p.signatures, g.signatures, (g.V) * sizeof(unsigned int) * Signature_Properties, cudaMemcpyHostToDevice));

    // incoming neighbors now
    // std::cout<<"4"<<std::endl;
    chkerr(cudaMalloc(&(p.incoming_neighbors), g.incoming_neighbors_offset[g.V] * sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.incoming_neighbors, g.incoming_neighbors, g.incoming_neighbors_offset[g.V] * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // std::cout<<"5"<<std::endl;
    chkerr(cudaMalloc(&(p.incoming_neighbors_offset), (g.V + 1) * sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.incoming_neighbors_offset, g.incoming_neighbors_offset, (g.V + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // attributes
    // std::cout<<"6"<<std::endl;
    chkerr(cudaMalloc(&(p.attributes), (g.V) * sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.attributes, g.attributes, (g.V) * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // std::cout<<"7"<<std::endl;
    chkerr(cudaMalloc(&(p.attributes_in_order), g.attributes_in_order_offset[g.num_attributes] * sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.attributes_in_order, g.attributes_in_order, g.attributes_in_order_offset[g.num_attributes] * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // std::cout<<"8"<<std::endl;
    chkerr(cudaMalloc(&(p.attributes_in_order_offset), (g.num_attributes + 1) * sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.attributes_in_order_offset, g.attributes_in_order_offset, (g.num_attributes + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));

    std::cout << "7" << std::endl;
    chkerr(cudaMalloc(&(p.two_hop_neighbors), g.two_hop_neighbors_offset[g.V] * sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.two_hop_neighbors, g.two_hop_neighbors, g.two_hop_neighbors_offset[g.V] * sizeof(unsigned int), cudaMemcpyHostToDevice));

    std::cout << "8" << std::endl;
    chkerr(cudaMalloc(&(p.two_hop_neighbors_offset), (g.V + 1) * sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.two_hop_neighbors_offset, g.two_hop_neighbors_offset, (g.V + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));

    std::cout << "9" << std::endl;
    p.V = g.V;

    p.E = g.E;

    // std::cout<<"10"<<std::endl;
    p.num_attributes = g.num_attributes;
}

void malloc_extra_to_gpu_memory(E_pointers &e, unsigned int v, unsigned int *v_order)
{

    // so cudaMemoryManaged allows cpu and gpu have access to memory at a cost (speed)
    // need this for our results table

    chkerr(cudaMalloc(&(e.matching_order), (v) * sizeof(unsigned int)));
    chkerr(cudaMemcpy(e.matching_order, v_order, (v) * sizeof(unsigned int), cudaMemcpyHostToDevice));

    chkerr(cudaMallocManaged(&(e.result_lengths), (v + 1) * sizeof(unsigned int)));

    // datastructure
    // this table_size is variable based on workers, but for now im setting it constant
    unsigned int table_size = 5000 chkerr(cudaMalloc(&(e.results_table), table_size));
    chkerr(cudaMalloc(&(e.indexes_table), table_size));
    chkerr(cudaMalloc(&(e.scores_table), table_size));
    chkerr(cudaMalloc(&(e.intra_v_table), table_size));

    chkerr(cudaMalloc(&(e.write_pos), sizeof(unsigned long long int)));
    chkerr(cudaMemset(e.write_pos, 0, sizeof(unsigned long long int)));

    unsigned int buffer_size = BUFFER_TABLE_SIZE * sizeof(unsigned int);
    chkerr(cudaMalloc(&(e.helper_buffer), buffer_size));
}