project(imageEnhancement)
set(PROJECT_NAME imageEnhancement)
add_compile_options(-Wall -fexec-charset=GBK -std=c++11)
include_directories(./include)
aux_source_directory(src SRC_DIR_VAR)
add_executable(${PROJECT_NAME} ${SRC_DIR_VAR})

target_link_libraries(${PROJECT_NAME} PARENT)
target_link_libraries(${PROJECT_NAME} PARENT_HEAD)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O3")

# add_custom_command(
#     OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/myscript.bat
#     COMMAND echo "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.exe" > ${CMAKE_CURRENT_BINARY_DIR}/myscript.bat
#     COMMENT "generating ${CMAKE_CURRENT_BINARY_DIR}/myscript.bat"
#     )
    
    
# add_custom_target(
#     run_${PROJECT_NAME} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/myscript.bat
#     COMMAND ${CMAKE_COMMAND} -E echo "running ${CMAKE_CURRENT_BINARY_DIR}/myscript.bat"
#     COMMAND
#          ./myscript.bat
#     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#     COMMENT "build ${CMAKE_CURRENT_BINARY_DIR}/myscript.bat"
# )
