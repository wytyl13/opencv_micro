project(PARENT_HEAD)
include_directories(./include)
aux_source_directory(src DIR_LIB_SRCS)
# generate the lib 
# notice the difference between static and dynamic lib.
# the former will be added into the exe file, and the last will not.
# so the former will be lower efficient at the peroid of compiling.
# and the last will be lower efficient at the peroid of running.
add_library(${PROJECT_NAME} STATIC ${DIR_LIB_SRCS})

# this code is important. you can add the head file in the include path in this parent program
# in the other module. and you need not set any code in the cmakeLists.txt file in other module.

target_include_directories(PARENT_HEAD PUBLIC include)
target_link_libraries(${PROJECT_NAME} PARENT)
