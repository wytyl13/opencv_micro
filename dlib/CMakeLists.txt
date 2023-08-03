#set the project_name is opencv for the variable PROJECT_NAME
project(dlib)

# this code can be ignored.
set(PROJECT_NAME dlib)

#add the param, it is dedicated to c and cpp compiler.
# -g: generate the exe file that support the gdb.
# -c: compiler the c/cpp file and generate the .o file.
# -S: just compiler, not assembly.
add_compile_options(-Wall -fexec-charset=GBK -std=c++11)
# set the include directories in the current module.
include_directories(./include)

# define the variable that stored all the cpp file
# involved exe file and other implement of the head file.
# of course, this method will make it easy to generate many
# exe files in one modules. we have used micro program method
# to implement the micro server like java in cpp program. it has
# made it easy to generate many module in one program. then, we 
# will generate many exe files in one module. of curse, it is not confilct
# for us to code one multi-function in one module. we should define one main function
# that can handle multi application scenarios as far as possible.

# notice, the SRC_DIR_VAR stored all the implement file of head file.
# then we should scan each cpp file that you want to generate one exe file used it in main directory.
# you should compiler all the file in src directory and each cpp file in main directory
# and generate the corresponding exe file. so you should define each exe file name based
# on each cpp file in main directory. but you should notice, this compiler efficient will
# be lower than the original program, because you will compiler many times.
aux_source_directory(src SRC_DIR_VAR)


set(DLIB_INCLUDE "D:\\development_app2\\dlib-19.24")
set(DLIB_LIBS "D:/development_app2/dlib-19.24/build/dlib/libdlib.a")
# set the compiler param that dedicated to using for cpp.
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O3")
include_directories(${DLIB_INCLUDE})
# Find all *.cpp files in main directory and store in list variables exe_cpp_list
file(GLOB MAIN_DIR_VAR main/*.cpp)
foreach(EXE_CPP_FILE IN LISTS MAIN_DIR_VAR)
    # get the cpp file name based on the file path.
    get_filename_component(EXE_CPP_FILE_NAME ${EXE_CPP_FILE} NAME_WE)
    # append each cpp file in main directory to the variable SRC_DIR_VAR
    set(ALL_SRC_CPP "${SRC_DIR_VAR};main/${EXE_CPP_FILE_NAME}.cpp")
    # compile each ALL_SRC_CPP and generate the exe file based on the file name of EXE_CPP_FILE_NAME
    # because each cpp file is not repeated, so you can generate the exe file based 
    # on each file name. we always generate the exe file based on the program name or module name before.
    # because we just need to generate one exe file in one program or one module.
    message(${EXE_CPP_FILE_NAME})
    # set the compiler command.
    add_executable(${EXE_CPP_FILE_NAME} ${ALL_SRC_CPP})
    # add the linked library for the current compiler exe file.
    # add the dlib library and parent library what involved some libraries what defined by yourself
    # involved parent library and parent-head library what are compilerd based on opencv library and
    # other library.
    target_link_libraries(${EXE_CPP_FILE_NAME} PARENT)
    target_link_libraries(${EXE_CPP_FILE_NAME} PARENT_HEAD)
    target_link_libraries(${EXE_CPP_FILE_NAME} ${DLIB_LIBS})
endforeach()






# of course, you can also add the sub directory, of course, you need to compiler and
# generate the dynamic or static lib for the subdirectory. the lib name is main.


# the compiler command


# because you have added the subDirectory, so you can reference the dynamic lib.
# target_link_libraries(${PROJECT_NAME} PARENT)
# target_link_libraries(${PROJECT_NAME} PARENT_HEAD)

# set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O3")

# set(DLIB_INCLUDE "D:\\development_app2\\dlib-19.24")
# set(DLIB_LIBS "D:/development_app2/dlib-19.24/build/dlib/libdlib.a")
# message(${DLIB_LIBS})
# include_directories(${DLIB_INCLUDE})
# target_link_libraries(${PROJECT_NAME} ${DLIB_LIBS})