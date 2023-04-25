#include "../inc/graph.h"
#include "../inc/main.h"


int main(int argc, char *argv[]) {
    
        // Start the clock
    auto start = std::chrono::high_resolution_clock::now();
    
    // Code to be timed
    // ...
    


    approx_searching(argv[1],argv[2],5u);

    cout<<"Finished Completely"<<endl;

    // End the clock
    auto end = std::chrono::high_resolution_clock::now();
    
    // Calculate the duration in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Time taken: " << duration << " ms" << std::endl;

    return 0;
}
