cmake_minimum_required(VERSION 3.5)

project(ModeloCPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#Se crea una variable para las fuentes.

set(project_sources
    main.cpp
    Extraction/extraction.cpp
    RegressionLogistic/regressionlogistic.cpp)

#Se crea una variable para los headers
set(project_headers
    Extraction/extraction.h
    RegressionLogistic/regressionlogistic.h)

#Se crean los ejecutables con todos los archivos fuentes necesarios
add_executable(${PROJECT_NAME}
    ${project_sources}
    ${project_headers})
