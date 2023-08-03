#set the project_name is opencv for the variable PROJECT_NAME
project(traditional)

# this code can be ignored.
set(PROJECT_NAME traditional)

#add the param
add_compile_options(-Wall -fexec-charset=GBK -std=c++11)
include_directories(./include)
aux_source_directory(src SRC_DIR_VAR)


# of course, you can also add the sub directory, of course, you need to compiler and
# generate the dynamic or static lib for the subdirectory. the lib name is main.


# the compiler command
add_executable(${PROJECT_NAME} ${SRC_DIR_VAR})

# becau2se you have added the subDirectory, so you can reference the dynamic lib.
target_link_libraries(${PROJECT_NAME} PARENT)
target_link_libraries(${PROJECT_NAME} PARENT_HEAD)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O3")
