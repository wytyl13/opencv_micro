project(PARENT)
include_directories(include)
aux_source_directory(src DIR_LIB_SRCS)
# generate the lib 
add_library(${PROJECT_NAME} STATIC ${DIR_LIB_SRCS})

# this code is important. you can add the head file in the include path in this parent program
# in the other module. and you need not set any code in the cmakeLists.txt file in other module.
# notice, this command will include the head file of the sub project into the project.
# so you need not include the head file in other sub project when you add the lib.
# so this command means you can include the head file of the sub project in other sub project.
# you need not add head file of this sub project in other sub project's cmakelists file used
# link_directories command.
target_include_directories(PARENT PUBLIC include)

set(OpenCV_DIR D:\\development_app2\\opencv\\build\\install)
set(OpenCV_INCLUDE_DIR D:\\development_app2\\opencv\\build\\install\\include)
set(OpenCV_Contrib "D:\\development_app2\\opencv\\source\\opencv_contrib\\modules")

set(EIGEN3_INCLUDE $ENV{EIGEN_HOME})
find_package(OpenCV REQUIRED)
if((OpenCV_FOUND))
    #add the include path
    message("WHOAMI")
    include_directories(${OpenCV_INCLUDE_DIR})
    include_directories(${EIGEN3_INCLUDE})
    include_directories(${OpenCV_Contrib})
    include_directories($ENV{PYTHON_HOME}\\include)
    #add the lib
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})


    message("have found opencv include: ${OpenCV_INCLUDE_DIR}")
    message("have found opencv libs: ${OpenCV_LIBS}")
    message("have found eigen include: ${EIGEN3_INCLUDE}")
    message("have found python include: $ENV{PYTHON_HOME}\\include")
else()
    message(FATAL_ERROR "not found opencv!")
endif()