#include "../inc/graph.h"
#include "../inc/main.h"

int main() {
    /**

    std::string input;
    std::string input2;
    std::cout << "Please enter a Query graph file: ";
    std::getline(std::cin, input);
    std::cout << "Please enter a Data graph file: ";
    std::getline(std::cin, input2);
    */

    approx_searching("test_dataset/query.format","test_dataset/data.txt",5u);

    return 0;
}