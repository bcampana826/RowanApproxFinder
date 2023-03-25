#include "../inc/memory.h"


inline void chkerr(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        std::cout<<cudaGetErrorString(code)<<std::endl;
        exit(-1);
    }
}

void malloc_graph_to_gpu_memory(Graph &g, G_pointers &p, bool query){

    chkerr(cudaMalloc(&(p.outgoing_neighbors),g.outgoing_neighbors_offset[g.V]*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.outgoing_neighbors,g.outgoing_neighbors,g.outgoing_neighbors_offset[g.V]*sizeof(unsigned int),cudaMemcpyHostToDevice));
   
    chkerr(cudaMalloc(&(p.outgoing_neighbors_offset),(g.V+1)*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.outgoing_neighbors_offset,g.outgoing_neighbors_offset,(g.V+1)*sizeof(unsigned int),cudaMemcpyHostToDevice));

    // signatures
    chkerr(cudaMalloc(&(p.signatures),(g.V)*sizeof(unsigned int)*Signature_Properties));
    chkerr(cudaMemcpy(p.signatures,g.signatures,(g.V)*sizeof(unsigned int)*Signature_Properties,cudaMemcpyHostToDevice));

    // incoming neighbors now
    chkerr(cudaMalloc(&(p.incoming_neighbors),g.incoming_neighbors_offset[g.V]*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.incoming_neighbors,g.incoming_neighbors,g.incoming_neighbors_offset[g.V]*sizeof(unsigned int),cudaMemcpyHostToDevice));

    chkerr(cudaMalloc(&(p.incoming_neighbors_offset),(g.V+1)*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.incoming_neighbors_offset,g.incoming_neighbors_offset,(g.V+1)*sizeof(unsigned int),cudaMemcpyHostToDevice));

    // attributes
    chkerr(cudaMalloc(&(p.attributes),(g.V)*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.attributes,g.attributes,(g.V)*sizeof(unsigned int),cudaMemcpyHostToDevice));

    chkerr(cudaMalloc(&(p.attributes_in_order),g.attributes_in_order_offset[g.largest_att+1]*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.attributes_in_order,g.attributes_in_order,g.attributes_in_order_offset[g.largest_att+1]*sizeof(unsigned int),cudaMemcpyHostToDevice));

    chkerr(cudaMalloc(&(p.attributes_in_order_offset),(g.V+1)*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.attributes_in_order_offset,g.attributes_in_order_offset,(g.V+1)*sizeof(unsigned int),cudaMemcpyHostToDevice));


    p.V = g.V;

    p.largest_att = g.largest_att;

}