#version 450

// получаем данные из вершинного шейдера
layout(location = 0) in vec3 f_worldPos;
layout(location = 1) in vec3 f_normal;
layout(location = 2) in vec2 f_uv;

// на выходе конечный цвет пикселя
layout(location = 0) out vec4 final_color;

layout(binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float _pad0;
    vec3 specular_color;
    float shininess;
} model_uniforms;

layout(binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    vec3 camera_position;
    float _pad0;
    uint spot_light_count;
    uint point_light_count;
    float _pad1[2];
} scene_uniforms;

layout(binding = 2, std140) uniform DirectionalLightUBO {
    vec3 direction;
    float _pad0;
    vec3 color;
    float _pad1;
    float intensity;
    float _pad2[3];
} dir_light;

// Storage Buffer для прожекторов
struct SpotLight {
    vec3 position;
    float _pad0;
    vec3 direction;
    float _pad1;
    vec3 color;
    float _pad2;
    float intensity;
    float inner_cos;
    float outer_cos;
    float _pad3;
};

layout(binding = 3, std430) readonly buffer SpotLightsSSBO {
    SpotLight spot_lights[];
};

// Storage Buffer для точечных источников
struct PointLight {
    vec3 position;
    float _pad0;
    vec3 color;
    float _pad1;
    float intensity;
    float radius;
    float _pad2[2];
};

layout(binding = 4, std430) readonly buffer PointLightsSSBO {
    PointLight point_lights[];
};

const float ambientStrength = 0.3;

// Универсальная функция Блинна-Фонга
vec3 calcBlinnPhong(vec3 N, vec3 L, vec3 V, vec3 lightColor, float intensity, 
                   vec3 albedo, vec3 specular, float shininess, float attenuation, float spotFactor) {
    
    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse_component = albedo * NdotL; 

    vec3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);
    vec3 specular_component = specular * pow(NdotH, shininess);  

    vec3 blinnPhong = (diffuse_component + specular_component) * NdotL * lightColor;
    return blinnPhong * intensity * attenuation * spotFactor;
}

// Направленный свет
vec3 calcDirectional(vec3 N, vec3 V, vec3 albedo, vec3 specular, float shininess) {
    // -dir_light.direction - направление к свету
	vec3 L = normalize(-dir_light.direction);
    
    return calcBlinnPhong(N, L, V, dir_light.color, dir_light.intensity, 
                         albedo, specular, shininess, 1.0, 1.0);
}

// Прожекторный свет
vec3 calcSpot(vec3 N, vec3 V, vec3 fragPos, SpotLight light, vec3 albedo, vec3 specular, float shininess) {
    vec3 toLight = light.position - fragPos;
    float dist = length(toLight);
    
    if (dist <= 0.0001) return vec3(0.0);
    
    vec3 L = normalize(toLight);
    vec3 spotDir = normalize(light.direction);
    
    // Косинус угла между направлением от точки к прожектору и направлением прожектора
    float cosTheta = dot(-L, spotDir);
    
    // light.outer_cos - угол вне конуса
    if (cosTheta < light.outer_cos) return vec3(0.0);
    
    // плавные края
    float spotFactor = 1.0;
    if (cosTheta < light.inner_cos) {
        spotFactor = (cosTheta - light.outer_cos) / (light.inner_cos - light.outer_cos);
    }
    
    float attenuation = 1.0 / (1.0 + 0.1 * dist * dist);
    
    return calcBlinnPhong(N, L, V, light.color, light.intensity, 
                         albedo, specular, shininess, attenuation, spotFactor);
}

// Точечный свет
vec3 calcPoint(vec3 N, vec3 V, vec3 fragPos, PointLight light, vec3 albedo, vec3 specular, float shininess) {
    // Вектор от точки к источнику света
	vec3 toLight = light.position - fragPos;
	// Расстояние до источника
    float dist = length(toLight);
    
    if (dist <= 0.0001 || dist > light.radius) return vec3(0.0);
    
    vec3 L = normalize(toLight);
    
    float attenuation = 1.0 / (1.0 + 0.1 * dist * dist);
    
    // Дополнительное затухание на границе радиуса
    float radiusFactor = 1.0 - smoothstep(light.radius * 0.7, light.radius, dist);
    attenuation *= radiusFactor;
    
    return calcBlinnPhong(N, L, V, light.color, light.intensity, 
                         albedo, specular, shininess, attenuation, 1.0);
}

//объединяем все типы освещения в единый цвет пикселя
void main() {
    vec3 N = normalize(f_normal);
    vec3 V = normalize(scene_uniforms.camera_position - f_worldPos);
    vec3 albedo = model_uniforms.albedo_color.rgb;
    vec3 specular = model_uniforms.specular_color;
    float shininess = model_uniforms.shininess;

    // Внешний свет
    vec3 color = ambientStrength * albedo;

    // Направленный свет
    color += calcDirectional(N, V, albedo, specular, shininess);

    // Прожекторы из storage buffer
    for (uint i = 0; i < scene_uniforms.spot_light_count; ++i) {
        color += calcSpot(N, V, f_worldPos, spot_lights[i], albedo, specular, shininess);
    }

    // Точечные источники из storage buffer
    for (uint i = 0; i < scene_uniforms.point_light_count; ++i) {
        color += calcPoint(N, V, f_worldPos, point_lights[i], albedo, specular, shininess);
    }

    color = color / (color + vec3(1.0));

    final_color = vec4(color, 1.0);
}