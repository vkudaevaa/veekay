#version 450

// из вершинного буфера передаём атрибуты вершины в вершинный шейдер
layout(location = 0) in vec3 v_position;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_uv;

// из вершинного шейдера во фрагментный идут координы вершины, позиция нормали и текстурные координы в мировом пространстве
layout(location = 0) out vec3 f_worldPos;
layout(location = 1) out vec3 f_normal;
layout(location = 2) out vec2 f_uv;
// Shadow coordinates (как в эталонном коде)
layout(location = 3) out vec4 f_dirLightSpacePos;
layout(location = 4) out vec4 f_spotLightSpacePos;

layout(binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    mat4 dir_light_matrix;
    mat4 spot_light_matrix;
    vec3 camera_position;
    float _pad0;
    uint spot_light_count;
};

layout(binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec4 albedo_color;
};

void main() {
    vec4 worldPos4 = model * vec4(v_position, 1.0);
    
    // ИСПРАВЛЕНИЕ: Используйте обратную транспонированную матрицу
    mat3 normalMatrix = transpose(inverse(mat3(model)));
    vec3 worldNormal = normalize(normalMatrix * v_normal);
    
    gl_Position = view_projection * worldPos4;
    
    f_worldPos = worldPos4.xyz;
    f_normal = worldNormal;  // Исправленная нормаль
    f_uv = v_uv;
    
    // Shadow coordinates
    f_dirLightSpacePos = dir_light_matrix * worldPos4;
    f_spotLightSpacePos = spot_light_matrix * worldPos4;
}