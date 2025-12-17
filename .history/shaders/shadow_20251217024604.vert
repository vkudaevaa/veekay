#version 450

// получаем позицию вершины из буфера вершин
layout(location = 0) in vec3 v_position;

layout(push_constant) uniform Push {
    mat4 light_view_proj;
} push;

layout(binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float _pad0;
    vec3 specular_color;
    float shininess;
};

void main() {
    gl_Position = push.light_view_proj * model * vec4(v_position, 1.0);
}