#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_for_each.hpp>
#include <hpx/include/parallel_execution.hpp>
#include <iostream>
#include <vector>

using Matrix = std::vector<std::vector<int>>;

// Function to perform parallel matrix multiplication
Matrix parallel_matrix_multiply(const Matrix& A, const Matrix& B, int size) {
    Matrix C(size, std::vector<int>(size, 0));

    // HPX Parallel For Loop across the rows
    hpx::ranges::for_each(hpx::execution::par, 
        hpx::util::counting_iterator<int>(0), 
        hpx::util::counting_iterator<int>(size), 
        [&](int i) {
            for (int j = 0; j < size; ++j) {
                for (int k = 0; k < size; ++k) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        });

    return C;
}

int hpx_main(hpx::program_options::variables_map& vm) {
    int size = 100; // 100x100 matrix

    // Initialize matrices with dummy data
    Matrix A(size, std::vector<int>(size, 1));
    Matrix B(size, std::vector<int>(size, 2));

    std::cout << "Starting HPX Parallel Matrix Multiplication (" << size << "x" << size << ")...\n";
    Matrix C = parallel_matrix_multiply(A, B, size);
    std::cout << "Computation complete. Sample result C[0][0]: " << C[0][0] << std::endl;

    return hpx::finalize();
}

int main(int argc, char* argv[]) {
    return hpx::init(argc, argv);
}
