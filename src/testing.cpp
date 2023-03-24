#include "../inc/graph.h"
#include "../inc/main.h"

int main() {
    

    std::string input;
    std::string input2;
    std::cout << "Please enter a Query graph file: ";
    std::getline(std::cin, input);
    std::cout << "Please enter a Data graph file: ";
    std::getline(std::cin, input2);
    

    //approx_searching(input,input2,5u);

    cout<<"start loading graph file from disk to memory..."<<endl;
    Graph query_graph(true,input);

    cout<<"Query Loaded!"<<endl;

    Graph data_graph(false,input2);

    cout<<"Loaded"<<endl;

    return 0;
}