cmake_minimum_required(VERSION 3.9)
project(OptCuts)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(OpenMP REQUIRED)
# libigl
option(LIBIGL_USE_STATIC_LIBRARY "Use libigl as static library" OFF)
option(LIBIGL_WITH_ANTTWEAKBAR       "Use AntTweakBar"    OFF)
option(LIBIGL_WITH_CGAL              "Use CGAL"           OFF)
option(LIBIGL_WITH_COMISO            "Use CoMiso"         OFF)
option(LIBIGL_WITH_CORK              "Use Cork"           OFF)
option(LIBIGL_WITH_EMBREE            "Use Embree"         OFF)
option(LIBIGL_WITH_LIM               "Use LIM"            OFF)
option(LIBIGL_WITH_MATLAB            "Use Matlab"         OFF)
option(LIBIGL_WITH_MOSEK             "Use MOSEK"          OFF)
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"          OFF)
option(LIBIGL_WITH_PNG               "Use PNG"            ON)
option(LIBIGL_WITH_PYTHON            "Use Python"         OFF)
option(LIBIGL_WITH_TETGEN            "Use Tetgen"         OFF)
option(LIBIGL_WITH_TRIANGLE          "Use Triangle"       ON)
option(LIBIGL_WITH_VIEWER            "Use OpenGL viewer"  ON)
option(LIBIGL_WITH_XML               "Use XML"            OFF)

find_package(LIBIGL REQUIRED QUIET)

# Check to see if PARDISO is available
find_library(PARDISO
  pardiso600-INTEL1301-X86-64
  pardiso600-GNU461-X86-64
  pardiso600-GNU472-X86-64
  pardiso600-GNU481-X86-64
  pardiso600-MPI-INTEL1301-X86-64
  pardiso600-MPI-GNU450-X86-64
  pardiso600-MPI-GNU461-X86-64
  pardiso600-MPI-GNU463-X86-64
  pardiso600-MPI-GNU472-X86-64
  pardiso600-WIN-X86-64
  pardiso600-MACOS-X86-64
  PATHS ${CMAKE_CURRENT_SOURCE_DIR}/ext/pardiso
)


include(ext/tbb/cmake/TBBGet.cmake)
tbb_get(TBB_ROOT tbb_root CONFIG_DIR TBB_DIR)
find_package(TBB REQUIRED tbb)

# brew install libomp
if(NOT TARGET OpenMP::OpenMP_CXX)
    add_library(OpenMP_TARGET INTERFACE)
    add_library(OpenMP::OpenMP_CXX ALIAS OpenMP_TARGET)
    target_compile_options(OpenMP_TARGET INTERFACE ${OpenMP_CXX_FLAGS})
    find_package(Threads REQUIRED)
    target_link_libraries(OpenMP_TARGET INTERFACE Threads::Threads)
    target_link_libraries(OpenMP_TARGET INTERFACE ${OpenMP_CXX_FLAGS})
endif()

file(GLOB SRCFILES FracCuts/*.cpp)
add_executable(${PROJECT_NAME}_bin ${SRCFILES})
target_link_libraries(${PROJECT_NAME}_bin igl::core igl::opengl_glfw igl::triangle igl::png ${PARDISO} TBB::tbb OpenMP::OpenMP_CXX)
