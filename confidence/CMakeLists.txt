cmake_minimum_required(VERSION 3.14)
project(_kdtree CXX)

set(CMAKE_CXX_STANDARD 17)

# Find pybind11 — use the venv python explicitly to avoid pyenv shim issues
set(_PYTHON_CANDIDATES
    "/bata/ahmealy/personal_projects/meshGraphNets_pytorch/venv/bin/python"
    "/home/ahmealy/ML/meshGraphNets_pytorch/venv/bin/python"
    "python3"
)
foreach(_PY IN LISTS _PYTHON_CANDIDATES)
    execute_process(
        COMMAND ${_PY} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _PY_RET
    )
    if(_PY_RET EQUAL 0 AND pybind11_DIR)
        message(STATUS "Found pybind11 via ${_PY}: ${pybind11_DIR}")
        break()
    endif()
endforeach()

find_package(pybind11 REQUIRED)

pybind11_add_module(_kdtree
    kdtree.cpp
    kdtree_bind.cpp
)

target_compile_options(_kdtree PRIVATE -O3)

# Output the .so file directly into the confidence/ directory
set_target_properties(_kdtree PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
