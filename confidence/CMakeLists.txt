cmake_minimum_required(VERSION 3.14)
project(_kdtree CXX)

set(CMAKE_CXX_STANDARD 17)

# Find pybind11 (installed via pip install pybind11)
execute_process(
    COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED)

pybind11_add_module(_kdtree
    kdtree.cpp
    kdtree_bind.cpp
)

target_compile_options(_kdtree PRIVATE -O3 -march=native)

# Output the .so file directly into the confidence/ directory
set_target_properties(_kdtree PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
