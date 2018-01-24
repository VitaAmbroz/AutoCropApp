cmake_minimum_required(VERSION 2.8)

project( autocrop )
 
find_package( OpenCV REQUIRED )
find_package( Threads )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )

# change your local PATH of VLFeat here!
set( VLFEAT_INCLUDE_DIRS "/home/vita/programovani/vlfeat" )

include_directories( ${OpenCV_INCLUDE_DIRS} ${VLFEAT_INCLUDE_DIRS} )

add_executable( autocrop "src/Main.cpp"
"src/AttentionBased.cpp"
"src/AutoCropVCBSCP.cpp"
"src/SalMapStentiford.cpp"
"src/SalMapMargolin.cpp" )

# change your local PATH of VLFeat here!
add_library( vlfeats /home/vita/programovani/vlfeat/vl/host.c
/home/vita/programovani/vlfeat/vl/random.c
/home/vita/programovani/vlfeat/vl/generic.c 
/home/vita/programovani/vlfeat/vl/slic.c )

target_link_libraries( autocrop ${OpenCV_LIBS} vlfeats pthread)
