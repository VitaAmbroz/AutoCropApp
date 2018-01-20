cmake_minimum_required(VERSION 2.8)

project( AutoCrop )

# Add libraries 
find_package( OpenCV REQUIRED )
#find_package( Boost REQUIRED )

include_directories(${Boost_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable( AutoCrop "src/Main.cpp" "src/AttentionBased.cpp" "src/AutoCropVCBSCP.cpp" "src/SalMapStentiford.cpp" "src/SalMapMargolin.cpp" )

target_link_libraries( AutoCrop ${OpenCV_LIBS} )
# ${Boost_LIBRARIES}
