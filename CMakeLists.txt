cmake_minimum_required( VERSION 3.26 )
project( g2o_curve_fitting )

set( CMAKE_BUILD_TYPE "Release" )
# set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
list(APPEND CMAKE_MODULE_PATH /usr/local/lib/cmake/SuiteSparse)
# find_package(CHOLMOD REQUIRED)
find_package(CHOLMOD REQUIRED)

find_package( G2O REQUIRED )
include_directories(
    ${CHOLMOD_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    "/usr/include/eigen3"
)

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_DIRS} )


SET(G2O_LIBS 
      g2o_cli 
      g2o_ext_freeglut_minimal 
      g2o_simulator 
      g2o_solver_slam2d_linear 
      g2o_types_icp 
      g2o_types_slam2d 
      g2o_core 
      g2o_interface 
      g2o_solver_csparse 
      g2o_solver_structure_only 
      g2o_types_sba 
      g2o_types_slam3d 
      g2o_csparse_extension 
      g2o_opengl_helper 
      g2o_solver_dense 
      g2o_stuff 
      g2o_types_sclam2d 
      g2o_viewer 
      g2o_parser 
      g2o_solver_pcg 
      g2o_types_data 
      g2o_types_sim3 
      cxsparse 
      g2o_solver_cholmod
)
add_executable( main main.cpp )

target_link_libraries( main
    ${OpenCV_LIBS}
    ${G2O_LIBS}
    ${CHOLMOD_LIBRARIES}
)
