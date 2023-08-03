project(ocr)
set(PROJECT_NAME ocr)
add_compile_options(-Wall -fexec-charset=GBK -std=c++11)
include_directories(./include)
aux_source_directory(src SRC_DIR_VAR)
add_executable(${PROJECT_NAME} ${SRC_DIR_VAR})

target_link_libraries(${PROJECT_NAME} PARENT)
target_link_libraries(${PROJECT_NAME} PARENT_HEAD)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O3")
set(Tesseract_INCLUDE_DIR D:\\development_app2\\tesseract\\include)
include_directories(${Tesseract_INCLUDE_DIR})
# appedn the dynamic library path
# link_directories(D:\\development_app2\\Tesseract-OCR)
# target_link_libraries(${PROJECT_NAME} tesseract-5)
