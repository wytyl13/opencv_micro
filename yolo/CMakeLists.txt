#set the project_name is opencv for the variable PROJECT_NAME
project(yolo)

# this code can be ignored.
set(PROJECT_NAME yolo)

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

# of course, you can also define the rule by yourself.
# add_custom_command(
#     #  you can output some file based on the command and generate the rule based on the file content.
#     # ${CMAKE_CURRENT_BINARY_DIR} means ./build/yolo/
#     OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/rulesByYourself.txt
#     COMMAND echo "./${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}/${PROJECT_NAME}.exe" > ${CMAKE_CURRENT_BINARY_DIR}/rulesByYourself.txt
#     COMMENT "Generating rulesByYourself.txt"
# )

# # define the target that depends on the custom command rulesByYourself
# add_custom_target(
#     run_${PROJECT_NAME} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/rulesByYourself.txt
#     COMMENT "build rulesByYourself.txt"
# )

# of course, you can generate the target that to run one exe file.
# notice these code as follow will be made when you input the make command in the build path.
# of course, you can also make the single target. but you make the out target will execute 
# all the target. of course, you can define the code that running the scripts file in the target defined by yourself.
# it means it will run the exe file directly after you input make command for the current program.
add_custom_command(
    # you should output one file first. but you can also define the target directly.
    # you should define Target used POST_BUILD, it means you should build the target first.
    # notice the taget need to distingush the case. but this command is not suitable for this case.
    # TARGET run_${PROJECT_NAME} POST_BUILD
    # you can run the exe file used a script file.
    # generate the script file, the content is run the exe file.

    # then let's descrip these two command as follow.
    # cmake will create the target first, then it realized that it should depends on
    # the ${CMAKE_CURRENT_BINARY_DIR}/myscript.sh, so it will execute the command add_custom_command
    # it is all the process about these two command. of course, you can define the add_custom_command
    # command used TARGET run_${PROJECT_NAME} POST_BUILD, it means this process will
    # run the target first and then execute the rest code defined in the add_custom_command command.
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/myscript.bat
    # generate the scripts, but you should notice you should not run the scripts file at here.
    # because what we want to get is we want to define the target about run the scripts.
    # so you should not run the scripts at here when you generate the makefile based on the cmakeLists.txt file.
    # you need not to add the ./ before the exe file in the script file.
    COMMAND echo "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.exe" > ${CMAKE_CURRENT_BINARY_DIR}/myscript.bat
    # notice the order, the COMMENT WILL BE print first, then is the COMMAND.
    
    COMMENT "generating ${CMAKE_CURRENT_BINARY_DIR}/myscript.bat"
    )
    
    
add_custom_target(
    run_${PROJECT_NAME} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/myscript.bat
    # notice, you should generate the script file before you run it.
    # so you can run the scripts in the target command. because the command will rollback
    # if the command failed to run.
    COMMAND ${CMAKE_COMMAND} -E echo "running ${CMAKE_CURRENT_BINARY_DIR}/myscript.bat"
    # you can ./script.bat file directly, you should use sh script.bat file in linux. this command will run any sh file.
    # enven thougth the sh file has not the execute permission.
    COMMAND
         ./myscript.bat
    # specific the work directory about the script file you want to run.
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "build ${CMAKE_CURRENT_BINARY_DIR}/myscript.bat"
)

# the process order is 
# 70 -> 71 -> 45~67(COMMENT FIRST, THEN COMMAND STANDARD PRINT, IF ONE LINE FAILED, ROLL BACK AND INTERCEPT THE PROCESS) 
# ->  75~81