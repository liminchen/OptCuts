# Prepare dependencies
#
# For each third-party library, if the appropriate target doesn't exist yet,
# download it via external project, and add_subdirectory to build it alongside
# this project.


# Download and update 3rd_party libraries
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
list(REMOVE_DUPLICATES CMAKE_MODULE_PATH)
include(OptCutsDownloadExternal)

################################################################################
# Required libraries
################################################################################

# libigl
#if(NOT TARGET igl)
#  download_libigl()
#  add_subdirectory(${OPTCUTS_EXTERNAL}/libigl EXCLUDE_FROM_ALL)
#endif()

# TBB
if(NOT TARGET TBB::tbb)
  download_tbb()
  set(TBB_BUILD_STATIC ON CACHE BOOL " " FORCE)
  set(TBB_BUILD_SHARED OFF CACHE BOOL " " FORCE)
  set(TBB_BUILD_TBBMALLOC OFF CACHE BOOL " " FORCE)
  set(TBB_BUILD_TBBMALLOC_PROXY OFF CACHE BOOL " " FORCE)
  set(TBB_BUILD_TESTS OFF CACHE BOOL " " FORCE)
  add_subdirectory(${OPTCUTS_EXTERNAL}/tbb EXCLUDE_FROM_ALL)
  add_library(TBB::tbb ALIAS tbb_static)
endif()

# AMGCL
#if(NOT TARGET amgcl::amgcl)
#  download_amgcl()
#  add_subdirectory(${OPTCUTS_EXTERNAL}/amgcl EXCLUDE_FROM_ALL)
#endif()
