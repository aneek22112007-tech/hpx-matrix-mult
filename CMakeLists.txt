cmake_minimum_required(VERSION 3.18)
project(HPX_Matrix_Multiplication CXX)

# Find HPX
find_package(HPX REQUIRED)

# Add the executable
add_executable(matrix_mult matrix_mult.cpp)

# Link HPX to the executable
target_link_libraries(matrix_mult PUBLIC HPX::hpx)
