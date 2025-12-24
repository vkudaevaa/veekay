cmake_minimum_required(VERSION 3.20)

project(testbed LANGUAGES C CXX)

add_executable(${PROJECT_NAME} main.cpp)

set_target_properties(${PROJECT_NAME} PROPERTIES 
    CXX_STANDARD_REQUIRED TRUE 
    CXX_STANDARD 20
)

find_package(Vulkan REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE veekay Vulkan::Vulkan)

if(MSVC)
    target_link_options(${PROJECT_NAME} PRIVATE /subsystem:windows /entry:mainCRTStartup)
endif()

find_program(GLSLC_FOUND glslc)
if(GLSLC_FOUND)
    set(_SHADER_BINARIES)

    macro(compile_shader SHADER_FILE)
        set(SHADER_SOURCE ${CMAKE_SOURCE_DIR}/shaders/${SHADER_FILE})
        set(SHADER_BINARY ${SHADER_FILE}.spv)
        set(SHADER_BINARY_PATH ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_BINARY})

        add_custom_command(
            OUTPUT ${SHADER_BINARY_PATH}
            COMMAND glslc ${SHADER_SOURCE} -o ${SHADER_BINARY_PATH}
            DEPENDS ${SHADER_SOURCE}
            COMMENT "Compiling ${SHADER_FILE} shader"
        )

        list(APPEND _SHADER_BINARIES ${SHADER_BINARY_PATH})
    endmacro()

    compile_shader(shader.vert)
    compile_shader(shader.frag)

    add_custom_target(shaders DEPENDS ${_SHADER_BINARIES})
    add_dependencies(${PROJECT_NAME} shaders)

    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shaders
        $<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders
    )
endif()
