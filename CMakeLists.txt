cmake_minimum_required(VERSION 2.8)

project( autocrop )

find_package( Boost REQUIRED COMPONENTS filesystem ) 
find_package( OpenCV REQUIRED )
find_package( Threads )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp" )

# change your local PATH of VLFeat here!
set( VLFEAT_INCLUDE_DIRS "vlfeat" )

include_directories( ${OpenCV_INCLUDE_DIRS} ${VLFEAT_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS}) 

add_executable( autocrop "src/Main.cpp"
"src/Arguments.cpp"
"src/AutocropStentiford.cpp"
"src/AutocropFang.cpp"
"src/AutocropSuh.cpp"
"src/SalMapStentiford.cpp"
"src/SalMapMargolin.cpp" 
"src/SalMapItti.cpp"
"src/CompositionModel.cpp")

# change your local PATH of VLFeat here!
add_library( vlfeats vlfeat/vl/host.c vlfeat/vl/random.c vlfeat/vl/generic.c vlfeat/vl/slic.c )

target_link_libraries( autocrop ${OpenCV_LIBS} ${Boost_LIBRARIES} vlfeats pthread)
