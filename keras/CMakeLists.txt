#set the project_name is opencv for the variable PROJECT_NAME
project(KERAS)

# this code can be ignored.
set(PROJECT_NAME KERAS)

#add the param
add_compile_options(-Wall -fexec-charset=GBK -std=c++11)
aux_source_directory(src SRC_DIR_VAR)


# of course, you can also add the sub directory, of course, you need to compiler and
# generate the dynamic or static lib for the subdirectory. the lib name is main.


# the compiler command
# this command will generate the exe file for the sub project.
add_executable(${PROJECT_NAME} ${SRC_DIR_VAR})

# because you have added the subDirectory, so you can reference the static lib what you have generated.
# this command can link multi dynamic libs. used space.
# of course, you can also use it to link the static libs. but the link_libraries is
# dedicated to linking the static libs.
target_link_libraries(${PROJECT_NAME} PARENT)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O3")

