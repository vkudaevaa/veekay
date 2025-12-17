#version 450

layout(location = 0) in vec3 f_worldPos;
layout(location = 1) in vec3 f_normal;
layout(location = 2) in vec2 f_uv;
layout(location = 3) in vec4 f_dirLightSpacePos;// позиция вершины преобразованная в пространство направленного света
layout(location = 4) in vec4 f_spotLightSpacePos;

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
    mat4 dir_light_matrix;
    mat4 spot_light_matrix;
    vec3 camera_position;
    float _pad0;
    uint spot_light_count;
    float _pad1[3];
} scene_uniforms;

layout(binding = 2, std140) uniform DirectionalLightUBO {
    vec3 direction;
    float _pad0;
    vec3 color;
    float _pad1;
    float intensity;
    float _pad2[3];
} dir_light;

// Time uniforms для анимации солнца
layout(binding = 8, std140) uniform TimeUniforms {
    float time;        
    float frequency;  
    float amplitude;  
    float speed;      
} time_uniforms;

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

layout(binding = 5) uniform sampler2D albedo_texture;
layout(binding = 9) uniform sampler2DShadow dirShadowMap;
layout(binding = 10) uniform sampler2DShadow spotShadowMap;

const float ambientStrength = 0.3;

// Функция для расчета теней
float calcShadow(vec4 lightSpacePos, sampler2DShadow shadowMap) {
    vec3 projCoords = lightSpacePos.xyz / lightSpacePos.w;
    projCoords.xy = projCoords.xy * 0.5 + 0.5;
    if(projCoords.z > 1.0 || projCoords.x < 0.0 || projCoords.x > 1.0 || projCoords.y < 0.0 || projCoords.y > 1.0) 
        return 0.0;
    float shadow = texture(shadowMap, projCoords);
    return 1.0 - shadow;
}

// Универсальная функция Блинна-Фонга (со shadow factor)
vec3 calcBlinnPhong(vec3 N, vec3 L, vec3 V, vec3 lightColor, float intensity, 
                   vec3 albedo, vec3 specular, float shininess, float attenuation, 
                   float spotFactor, float shadowFactor) {
    
    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse_component = albedo * NdotL; 

    vec3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);
    vec3 specular_component = specular * pow(NdotH, shininess);  

    vec3 blinnPhong = (diffuse_component + specular_component) * NdotL * lightColor;
    return blinnPhong * intensity * attenuation * spotFactor * (1.0 - shadowFactor);
}

// Направленный свет (с тенью)
vec3 calcDirectional(vec3 N, vec3 V, vec3 albedo, vec3 specular, float shininess, float shadowFactor) {
    vec3 L = normalize(-dir_light.direction);
    
    return calcBlinnPhong(N, L, V, dir_light.color, dir_light.intensity, 
                         albedo, specular, shininess, 1.0, 1.0, shadowFactor);
}

// Прожекторный свет (с тенью)
vec3 calcSpot(vec3 N, vec3 V, vec3 fragPos, SpotLight light, vec3 albedo, vec3 specular, float shininess, float shadowFactor) {
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
                         albedo, specular, shininess, attenuation, spotFactor, shadowFactor);
}

void main() {
    // Нормализуем нормаль
    vec3 N = normalize(f_normal);
    vec3 V = normalize(scene_uniforms.camera_position - f_worldPos);
    
    vec2 uv = f_uv;

    if (abs(model_uniforms.shininess - 128.0) < 0.1) {
        // 1. Простая пульсация масштаба
        float pulse = sin(time_uniforms.time * time_uniforms.frequency) * 
                     time_uniforms.amplitude * 0.03;
        uv = (uv - 0.5) * (1.0 + pulse) + 0.5;
 
        vec2 center_vec = uv - 0.5;
        float dist_from_center = length(center_vec);

        float wave = sin(dist_from_center * 15.0 - time_uniforms.time * time_uniforms.speed) * 
                    time_uniforms.amplitude * 0.02;
        
        uv += normalize(center_vec) * wave;
        
        float rotation_angle = sin(time_uniforms.time * 0.3) * 0.05;
        mat2 rotation = mat2(
            cos(rotation_angle), -sin(rotation_angle),
            sin(rotation_angle), cos(rotation_angle)
        );
        uv = rotation * (uv - 0.5) + 0.5;
    }

    vec3 tex_color = texture(albedo_texture, uv).rgb;

    vec3 safe_tex = clamp(tex_color, vec3(0.001), vec3(0.999));
    tex_color = safe_tex / (vec3(1.0) - safe_tex);

    vec3 albedo = tex_color * model_uniforms.albedo_color;
    vec3 specular = model_uniforms.specular_color;
    float shininess = model_uniforms.shininess;

    float sDir = calcShadow(f_dirLightSpacePos, dirShadowMap);
    float sSpot = calcShadow(f_spotLightSpacePos, spotShadowMap);

    vec3 color = ambientStrength * albedo;

    color += calcDirectional(N, V, albedo, specular, shininess, sDir);

    for (uint i = 0; i < scene_uniforms.spot_light_count; ++i) {
        color += calcSpot(N, V, f_worldPos, spot_lights[i], albedo, specular, shininess, sSpot);
    }

    color = color / (color + vec3(1.0));

    final_color = vec4(color, 1.0);
}