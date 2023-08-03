# you can create the include directory in the lib program and you need not to create
# it in the main module that you want to generate the exe files based on the module.
# and you can create many lib module and many exe module.

cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 11)

# notice, this cmakeList.txt will generate the target OPENCV_MICRO, this OPENCV_MICRO
# involved all the program as follow. PARENT, PARENT_HEAD, KERAS, yolo.
# the dependency module just like the head module what you will generate the lib used them in other module
# and will not generate the exe file based on them.
# if you make OPENCV_MICRO, the outermost program. it invovled all the module.
# it will create all the target for all the module and generate the lib for head module and
# generate the exe file for the other module what you want to generate the exe file for them.
# and you can define the target by yourself in other module. these target you have defined by yourself
# will be generated at the same time when the program generated the target for the module that
# the target you have defined in one specific module. notice this content.
project(OPENCV_MICRO)

# notice, these code are dedicated to adding the sub directory for the current project.
# so these parameters inside of add_subdirectory should be same as the sub directory in the
# current project and it is case-insensitive. notice these parameters are not the project name
# you have defined in the cmakelists.txt what in each sub directory.
# and you should order it as what you want. because it will compiler as the order.
add_subdirectory(PARENT)
add_subdirectory(PARENT_HEAD)


# notice, you should place the program name what you want to generate the exe file
# at last.
# these code as follow will generate corrsponding exe file in build directory.
# you can only cmake this cmakeLists.txt file and make in build path. you will generate
# all the exe file as follow program.
add_subdirectory(KERAS)
add_subdirectory(imageEnhancement)
add_subdirectory(dlib)
add_subdirectory(frequency_domain_filtering)
add_subdirectory(traditional)
add_subdirectory(ocr)
add_subdirectory(yolo)

# of course, if you want to generate the exe file based on here. you should 
# link all the libs you have generated if you will use it.
# but we have not defined the exe file at here, this cmakeLists just as an 
# manager file, so we need not to add the libs.