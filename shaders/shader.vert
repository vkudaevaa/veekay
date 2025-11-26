#version 450

// из вершинного буфера передаём атрибуты вершины в вершинный шейдер
layout(location = 0) in vec3 v_position;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_uv;

// из вершинного шейдера во фрагментный идут координы вершины, позиция нормали и текстурные координы в мировом пространстве
layout(location = 0) out vec3 f_worldPos;
layout(location = 1) out vec3 f_normal;
layout(location = 2) out vec2 f_uv;

layout(binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    vec4 camera_pos;
};

layout(binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec4 albedo_color;
};

void main() {
    vec4 worldPos4 = model * vec4(v_position, 1.0);
    vec3 worldNormal = normalize(mat3(model) * v_normal);

	// преобразуем из мировых координат в координаты камеры и применяем перспективу
    gl_Position = view_projection * worldPos4;

	
    f_worldPos = worldPos4.xyz;
    f_normal = worldNormal;
    f_uv = v_uv;
}