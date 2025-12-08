#include <cstdint>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include <lodepng.h> // Для загрузки PNG

#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>

#include <chrono>

namespace {

    constexpr uint32_t max_models = 1024;
    constexpr uint32_t max_spot_lights = 16;
    constexpr uint32_t max_point_lights = 16;

    struct Vertex {
        veekay::vec3 position;
        veekay::vec3 normal;
        veekay::vec2 uv;
    };

    // для анимации текстуры солнца
    struct TimeUniforms {
        float time;
        float frequency = 2.0f;  
        float amplitude = 1.0f;  
        float speed = 0.2f;      
    };

    struct SceneUniforms {
        veekay::mat4 view_projection;
        veekay::vec3 camera_position;
        float _pad0;
        uint32_t spot_light_count;
        uint32_t point_light_count;
        float _pad1[2];
    };

    struct ModelUniforms {
        veekay::mat4 model;
        veekay::vec3 albedo_color;
        float _pad0;
        veekay::vec3 specular_color;
        float shininess;
    };

    struct DirectionalLightUBO {
        veekay::vec3 direction;
        float _pad0;
        veekay::vec3 color;
        float _pad1;
        float intensity;
        float _pad2[3];
    };

    struct SpotLightSSBO {
        veekay::vec3 position;
        float _pad0;
        veekay::vec3 direction;
        float _pad1;
        veekay::vec3 color;
        float _pad2;
        float intensity;
        float inner_cos;
        float outer_cos;
        float _pad3;
    };

    struct PointLightSSBO {
        veekay::vec3 position;
        float _pad0;
        veekay::vec3 color;
        float _pad1;
        float intensity;
        float radius;
        float _pad2[2];
    };

    struct Mesh {
        veekay::graphics::Buffer *vertex_buffer;
        veekay::graphics::Buffer *index_buffer;
        uint32_t indices;
    };

    struct Transform {
        veekay::vec3 position = {};
        veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
        veekay::vec3 rotation = {};

        veekay::mat4 matrix() const;
    };

    struct Model {
        Mesh mesh;
        Transform transform;
        veekay::vec3 albedo_color;
        veekay::vec3 specular_color;
        float shininess;
        uint32_t descriptor_set_index; // Индекс в массиве дескрипторных наборов
    };

    struct Camera {
        constexpr static float default_fov = 60.0f;
        constexpr static float default_near_plane = 0.01f;
        constexpr static float default_far_plane = 100.0f;

        veekay::vec3 position = {};
        veekay::vec3 rotation = {};

        float fov = default_fov;
        float near_plane = default_near_plane;
        float far_plane = default_far_plane;

        veekay::mat4 view() const;
        veekay::mat4 view_projection(float aspect_ratio) const;
    };

    inline namespace {
        Camera camera{
                .position = {-0.56f, -6.85f, -8.80f},
                .rotation = {-0.59f, -0.01f, 0.0f},
        };

        std::vector<Model> models;
        uint32_t spot_light_count = 1;
        uint32_t point_light_count = 2;
        
        float sun_rotation_angle = 0.0f;
        float earth_orbit_angle = 0.0f;
        float earth_rotation_angle = 0.0f;
        
        veekay::vec3 earth_position;
    }

    inline namespace {
        VkShaderModule vertex_shader_module = VK_NULL_HANDLE;
        VkShaderModule fragment_shader_module = VK_NULL_HANDLE;

        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> descriptor_sets; // Множество дескрипторных наборов

        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;

        veekay::graphics::Buffer *scene_uniforms_buffer = nullptr;
        veekay::graphics::Buffer *model_uniforms_buffer = nullptr;
        veekay::graphics::Buffer *directional_light_buffer = nullptr;
        veekay::graphics::Buffer *spot_lights_buffer = nullptr;
        veekay::graphics::Buffer *point_lights_buffer = nullptr;
        veekay::graphics::Buffer *time_uniforms_buffer = nullptr;

        Mesh plane_mesh;
        Mesh base_sphere_mesh;

        // Указатели на текстуры и сэмплеры для разных материалов
        veekay::graphics::Texture *missing_texture = nullptr;
        VkSampler missing_texture_sampler = VK_NULL_HANDLE;

        veekay::graphics::Texture *sun_texture = nullptr;
        VkSampler sun_sampler = VK_NULL_HANDLE;

        veekay::graphics::Texture *earth_texture = nullptr;
        VkSampler earth_sampler = VK_NULL_HANDLE;

        veekay::graphics::Texture *sky_texture = nullptr;
        VkSampler sky_sampler = VK_NULL_HANDLE;
    }

    float toRadians(float degrees) {
        return degrees * float(M_PI) / 180.0f;
    }

    veekay::mat4 Transform::matrix() const {
        auto t = veekay::mat4::translation(position);
        auto rx = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, toRadians(rotation.x));
        auto ry = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, toRadians(rotation.y));
        auto rz = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, toRadians(rotation.z));
        auto r = rz * ry * rx;
        auto s = veekay::mat4::scaling(scale);
        return t * r * s;
    }

    veekay::mat4 Camera::view() const {
        auto t = veekay::mat4::translation(-position);
        auto rx = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, -rotation.x);
        auto ry = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, -rotation.y);
        auto rz = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, -rotation.z);
        auto r = t * rz * ry * rx;
        return r;
    }

    veekay::mat4 Camera::view_projection(float aspect_ratio) const {
        auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
        return view() * projection;
    }

    VkShaderModule loadShaderModule(const char *path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.good()) {
            return VK_NULL_HANDLE;
        }
        size_t size = static_cast<size_t>(file.tellg());
        if (size == 0) return VK_NULL_HANDLE;
        std::vector<uint32_t> buffer((size + 3) / sizeof(uint32_t));
        file.seekg(0);
        file.read(reinterpret_cast<char *>(buffer.data()), size);
        file.close();

        VkShaderModuleCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .codeSize = buffer.size() * sizeof(uint32_t),
                .pCode = buffer.data(),
        };

        VkShaderModule result;
        if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS) {
            return VK_NULL_HANDLE;
        }

        return result;
    }

    Mesh createSphereMesh(float radius = 0.5f, uint32_t segments = 32) {
        Mesh sphere_mesh;
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;

        for (uint32_t lat = 0; lat <= segments; ++lat) {
            float theta = lat * float(M_PI) / segments;
            float sinTheta = std::sin(theta);
            float cosTheta = std::cos(theta);

            for (uint32_t lon = 0; lon <= segments; ++lon) {
                float phi = lon * 2.0f * float(M_PI) / segments;
                float sinPhi = std::sin(phi);
                float cosPhi = std::cos(phi);

                Vertex v;
                v.position.x = radius * sinTheta * cosPhi;
                v.position.y = radius * cosTheta;
                v.position.z = radius * sinTheta * sinPhi;
                v.normal = veekay::vec3::normalized(v.position);
                v.uv.x = float(lon) / segments;
                v.uv.y = float(lat) / segments;
                vertices.push_back(v);
            }
        }

        for (uint32_t lat = 0; lat < segments; ++lat) {
            for (uint32_t lon = 0; lon < segments; ++lon) {
                uint32_t first = lat * (segments + 1) + lon;
                uint32_t second = first + segments + 1;

                indices.push_back(first);
                indices.push_back(second);
                indices.push_back(first + 1);

                indices.push_back(second);
                indices.push_back(second + 1);
                indices.push_back(first + 1);
            }
        }

        sphere_mesh.vertex_buffer = new veekay::graphics::Buffer(
            vertices.size() * sizeof(Vertex), vertices.data(),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

        sphere_mesh.index_buffer = new veekay::graphics::Buffer(
            indices.size() * sizeof(uint32_t), indices.data(),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

        sphere_mesh.indices = uint32_t(indices.size());

        return sphere_mesh;
    }

    // Загрузка текстуры из файла (по аналогии с кодом из лекции)
    veekay::graphics::Texture* loadTextureFromFile(const char* path, VkCommandBuffer cmd) {
        std::vector<unsigned char> image_data; // вектор байтов, где каждые 4 байта образуют пиксель
        unsigned width = 0, height = 0;
        // преобразуем из png формата в вектор байтов
        unsigned error = lodepng::decode(image_data, width, height, path);

        if (error) {
            std::cerr << "Failed to load texture " << path << ": " << lodepng_error_text(error) << "\n";
            return nullptr;
        }

        // Конвертация RGBA в BGRA (так как PNG использует RGBA порядок, Vulkan ожидает BGRA)
        std::vector<uint32_t> bgra_data(width * height);
        for (size_t i = 0; i < width * height; ++i) {
            uint8_t r = image_data[i * 4 + 0];
            uint8_t g = image_data[i * 4 + 1];
            uint8_t b = image_data[i * 4 + 2];
            uint8_t a = image_data[i * 4 + 3];
            bgra_data[i] = (static_cast<uint32_t>(a) << 24) |
                           (static_cast<uint32_t>(r) << 16) |
                           (static_cast<uint32_t>(g) << 8) |
                           static_cast<uint32_t>(b);
        }

        return new veekay::graphics::Texture(cmd, width, height,
                                            VK_FORMAT_B8G8R8A8_UNORM,
                                            bgra_data.data());
    }

    // Создание сэмплера для сфер (Солнце, Земля)
    VkSampler createSphereSampler() {
        VkSamplerCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_LINEAR,
            .minFilter = VK_FILTER_LINEAR,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
            .mipLodBias = 0.0f,
            .anisotropyEnable = VK_TRUE,
            .maxAnisotropy = 16.0f,
            .compareEnable = VK_FALSE,
            .compareOp = VK_COMPARE_OP_ALWAYS,
            .minLod = 0.0f,
            .maxLod = VK_LOD_CLAMP_NONE,
            .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
            .unnormalizedCoordinates = VK_FALSE,
        };

        VkSampler sampler;
        if (vkCreateSampler(veekay::app.vk_device, &info, nullptr, &sampler) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan sphere texture sampler\n";
            return VK_NULL_HANDLE;
        }

        return sampler;
    }

    // Создание сэмплера для плоскости
    VkSampler createPlaneSampler() {
        VkSamplerCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_LINEAR,
            .minFilter = VK_FILTER_LINEAR,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
            .mipLodBias = 0.0f,
            .anisotropyEnable = VK_TRUE,
            .maxAnisotropy = 16.0f,
            .compareEnable = VK_FALSE,
            .compareOp = VK_COMPARE_OP_ALWAYS,
            .minLod = 0.0f,
            .maxLod = VK_LOD_CLAMP_NONE,
            .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
            .unnormalizedCoordinates = VK_FALSE,
        };

        VkSampler sampler;
        if (vkCreateSampler(veekay::app.vk_device, &info, nullptr, &sampler) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan plane texture sampler\n";
            return VK_NULL_HANDLE;
        }

        return sampler;
    }

    void initialize(VkCommandBuffer cmd) {
        VkDevice &device = veekay::app.vk_device;
        VkPhysicalDevice &physical_device = veekay::app.vk_physical_device;

        { 
            vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
            if (vertex_shader_module == VK_NULL_HANDLE) {
                std::cerr << "Failed to load Vulkan vertex shader from file\n";
                veekay::app.running = false;
                return;
            }

            fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
            if (fragment_shader_module == VK_NULL_HANDLE) {
                std::cerr << "Failed to load Vulkan fragment shader from file\n";
                veekay::app.running = false;
                return;
            }

            VkPipelineShaderStageCreateInfo stage_infos[2] = {
                {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_VERTEX_BIT,
                    .module = vertex_shader_module,
                    .pName = "main",
                },
                {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                    .module = fragment_shader_module,
                    .pName = "main",
                }
            };

            VkVertexInputBindingDescription buffer_binding{
                    .binding = 0,
                    .stride = sizeof(Vertex),
                    .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            };

            VkVertexInputAttributeDescription attributes[] = {
                {
                    .location = 0,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(Vertex, position),
                },
                {
                    .location = 1,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(Vertex, normal),
                },
                {
                    .location = 2,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32_SFLOAT,
                    .offset = offsetof(Vertex, uv),
                },
            };

            VkPipelineVertexInputStateCreateInfo input_state_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                    .vertexBindingDescriptionCount = 1,
                    .pVertexBindingDescriptions = &buffer_binding,
                    .vertexAttributeDescriptionCount = static_cast<uint32_t>(sizeof(attributes) / sizeof(attributes[0])),
                    .pVertexAttributeDescriptions = attributes,
            };

            VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                    .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            };

            VkPipelineRasterizationStateCreateInfo raster_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                    .polygonMode = VK_POLYGON_MODE_FILL,
                    .cullMode = VK_CULL_MODE_BACK_BIT,
                    .frontFace = VK_FRONT_FACE_CLOCKWISE,
                    .lineWidth = 1.0f,
            };

            VkPipelineMultisampleStateCreateInfo sample_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                    .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
                    .sampleShadingEnable = false,
                    .minSampleShading = 1.0f,
            };

            VkViewport viewport{
                    .x = 0.0f,
                    .y = 0.0f,
                    .width = static_cast<float>(veekay::app.window_width),
                    .height = static_cast<float>(veekay::app.window_height),
                    .minDepth = 0.0f,
                    .maxDepth = 1.0f,
            };

            VkRect2D scissor{
                    .offset = {0, 0},
                    .extent = {veekay::app.window_width, veekay::app.window_height},
            };

            VkPipelineViewportStateCreateInfo viewport_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                    .viewportCount = 1,
                    .pViewports = &viewport,
                    .scissorCount = 1,
                    .pScissors = &scissor,
            };

            VkPipelineDepthStencilStateCreateInfo depth_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                    .depthTestEnable = true,
                    .depthWriteEnable = true,
                    .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
            };

            VkPipelineColorBlendAttachmentState attachment_info{
                    .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                    VK_COLOR_COMPONENT_G_BIT |
                                    VK_COLOR_COMPONENT_B_BIT |
                                    VK_COLOR_COMPONENT_A_BIT,
            };

            VkPipelineColorBlendStateCreateInfo blend_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                    .logicOpEnable = false,
                    .logicOp = VK_LOGIC_OP_COPY,
                    .attachmentCount = 1,
                    .pAttachments = &attachment_info
            };

            {
                VkDescriptorPoolSize pools[] = {
                    {
                        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        .descriptorCount = 12,
                    },
                    {
                        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                        .descriptorCount = 8,
                    },
                    {
                        .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, // пул теперь резервирует место 
                        .descriptorCount = 16,                             // под новый тип дескрипторов
                    },
                    {
                        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        .descriptorCount = 16,
                    }
                };

                VkDescriptorPoolCreateInfo info{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                        .maxSets = 8, // количество наборов дескрипторов
                        .poolSizeCount = static_cast<uint32_t>(sizeof(pools) / sizeof(pools[0])),
                        .pPoolSizes = pools,
                };

                if (vkCreateDescriptorPool(device, &info, nullptr, &descriptor_pool) != VK_SUCCESS) {
                    std::cerr << "Failed to create Vulkan descriptor pool\n";
                    veekay::app.running = false;
                    return;
                }
            }

            // Layout дескрипторов
            {
                VkDescriptorSetLayoutBinding bindings[] = {
                    {
                        .binding = 0,
                        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    },
                    {
                        .binding = 1,
                        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    },
                    {
                        .binding = 2,
                        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                    },
                    {
                        .binding = 3,
                        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                    },
                    {
                        .binding = 4,
                        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                    },
                    {
                        .binding = 5, // привязка текстуры и сэмплера к определённому binding'у в шейдере
                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                    },
                    
                    {
                        .binding = 8, // для параметров времени
                        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                    },
                };

                VkDescriptorSetLayoutCreateInfo info{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                        .bindingCount = static_cast<uint32_t>(sizeof(bindings) / sizeof(bindings[0])),
                        .pBindings = bindings,
                };

                if (vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
                    std::cerr << "Failed to create Vulkan descriptor set layout\n";
                    veekay::app.running = false;
                    return;
                }
            }

            VkPipelineLayoutCreateInfo layout_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                    .setLayoutCount = 1,
                    .pSetLayouts = &descriptor_set_layout,
            };

            if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan pipeline layout\n";
                veekay::app.running = false;
                return;
            }

            VkGraphicsPipelineCreateInfo info{
                    .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                    .stageCount = 2,
                    .pStages = stage_infos,
                    .pVertexInputState = &input_state_info,
                    .pInputAssemblyState = &assembly_state_info,
                    .pViewportState = &viewport_info,
                    .pRasterizationState = &raster_info,
                    .pMultisampleState = &sample_info,
                    .pDepthStencilState = &depth_info,
                    .pColorBlendState = &blend_info,
                    .layout = pipeline_layout,
                    .renderPass = veekay::app.vk_render_pass,
            };

            if (vkCreateGraphicsPipelines(device, nullptr, 1, &info, nullptr, &pipeline) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan pipeline\n";
                veekay::app.running = false;
                return;
            }
        }

        scene_uniforms_buffer = new veekay::graphics::Buffer(
                sizeof(SceneUniforms),
                nullptr,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        model_uniforms_buffer = new veekay::graphics::Buffer(
                max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)),
                nullptr,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        directional_light_buffer = new veekay::graphics::Buffer(
                sizeof(DirectionalLightUBO),
                nullptr,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        spot_lights_buffer = new veekay::graphics::Buffer(
                max_spot_lights * sizeof(SpotLightSSBO),
                nullptr,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        point_lights_buffer = new veekay::graphics::Buffer(
                max_point_lights * sizeof(PointLightSSBO),
                nullptr,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        // будет содержать структуру TimeUniforms для управления анимацией        
        time_uniforms_buffer = new veekay::graphics::Buffer(
                sizeof(TimeUniforms),
                nullptr,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        // создаем текстуру по умолчанию, если основная не загрузилась
        {
            VkSamplerCreateInfo info{
                    .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                    .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            };

            if (vkCreateSampler(device, &info, nullptr, &missing_texture_sampler) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan texture sampler\n";
                veekay::app.running = false;
                return;
            }

            uint32_t pixels[] = {0xff000000, 0xffff00ff, 0xffff00ff, 0xff000000};
            missing_texture = new veekay::graphics::Texture(cmd, 2, 2, VK_FORMAT_B8G8R8A8_UNORM, pixels);
        }

        // загружаем реальные текстуры
        std::cout << "Loading textures...\n";
        
        sun_texture = loadTextureFromFile("./assets/sun.png", cmd);
        if (!sun_texture) {
            std::cerr << "Using missing texture for sun\n";
            sun_texture = missing_texture;
        }
        sun_sampler = createSphereSampler();
        if (sun_sampler == VK_NULL_HANDLE) sun_sampler = missing_texture_sampler;

        earth_texture = loadTextureFromFile("./assets/earth.png", cmd);
        if (!earth_texture) {
            std::cerr << "Using missing texture for earth\n";
            earth_texture = missing_texture;
        }
        earth_sampler = createSphereSampler();
        if (earth_sampler == VK_NULL_HANDLE) earth_sampler = missing_texture_sampler;

        sky_texture = loadTextureFromFile("./assets/sky.png", cmd);
        if (!sky_texture) {
            std::cerr << "Using missing texture for sky\n";
            sky_texture = missing_texture;
        }
        sky_sampler = createPlaneSampler(); 
        if (sky_sampler == VK_NULL_HANDLE) sky_sampler = missing_texture_sampler;

        {
            float plane_size = 25.0f;
            std::vector<Vertex> vertices = {
                {{-plane_size, 0.0f, plane_size},  {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}}, // повторяем 5 раз для детализации
                {{plane_size,  0.0f, plane_size},  {0.0f, -1.0f, 0.0f}, {5.0f, 0.0f}}, 
                {{plane_size,  0.0f, -plane_size}, {0.0f, -1.0f, 0.0f}, {5.0f, 5.0f}},
                {{-plane_size, 0.0f, -plane_size}, {0.0f, -1.0f, 0.0f}, {0.0f, 5.0f}},
            };

            std::vector<uint32_t> indices = {0, 1, 2, 2, 3, 0};

            plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
                    vertices.size() * sizeof(Vertex), vertices.data(),
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

            plane_mesh.index_buffer = new veekay::graphics::Buffer(
                    indices.size() * sizeof(uint32_t), indices.data(),
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

            plane_mesh.indices = uint32_t(indices.size());
        }

        base_sphere_mesh = createSphereMesh(0.5f, 32);

        // Создаем несколько дескрипторных наборов для разных материалов
        // 0 - солнце, 1 - земля, 2 - небо (плоскость), 3 - запасной
        descriptor_sets.resize(4);

        for (size_t i = 0; i < descriptor_sets.size(); ++i) {
            VkDescriptorSetAllocateInfo alloc_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = descriptor_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &descriptor_set_layout,
            };

            if (vkAllocateDescriptorSets(device, &alloc_info, &descriptor_sets[i]) != VK_SUCCESS) {
                std::cerr << "Failed to allocate descriptor set " << i << "\n";
                veekay::app.running = false;
                return;
            }

            // Выбираем текстуру и сэмплер для этого дескрипторного набора
            VkSampler sampler_to_use = missing_texture_sampler;
            veekay::graphics::Texture* texture_to_use = missing_texture;
            
            switch (i) {
                case 0: sampler_to_use = sun_sampler; texture_to_use = sun_texture; break;
                case 1: sampler_to_use = earth_sampler; texture_to_use = earth_texture; break;
                case 2: sampler_to_use = sky_sampler; texture_to_use = sky_texture; break;
                default: break; // Используем missing texture
            }

            VkDescriptorBufferInfo buffer_infos[] = {
                {
                    .buffer = scene_uniforms_buffer->buffer,
                    .offset = 0,
                    .range = sizeof(SceneUniforms),
                },
                {
                    .buffer = model_uniforms_buffer->buffer,
                    .offset = 0,
                    .range = sizeof(ModelUniforms),
                },
                {
                    .buffer = directional_light_buffer->buffer,
                    .offset = 0,
                    .range = sizeof(DirectionalLightUBO),
                },
                {
                    .buffer = spot_lights_buffer->buffer,
                    .offset = 0,
                    .range = max_spot_lights * sizeof(SpotLightSSBO),
                },
                {
                    .buffer = point_lights_buffer->buffer,
                    .offset = 0,
                    .range = max_point_lights * sizeof(PointLightSSBO),
                },
                    
                {
                    .buffer = time_uniforms_buffer->buffer,  // ДОБАВИТЬ
                    .offset = 0,
                    .range = sizeof(TimeUniforms),
                },
            };

            VkDescriptorImageInfo image_info{
                .sampler = sampler_to_use,
                .imageView = texture_to_use->view,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };

            VkWriteDescriptorSet write_infos[] = {
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor_sets[i],
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .pBufferInfo = &buffer_infos[0],
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor_sets[i],
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                    .pBufferInfo = &buffer_infos[1],
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor_sets[i],
                    .dstBinding = 2,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .pBufferInfo = &buffer_infos[2],
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor_sets[i],
                    .dstBinding = 3,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &buffer_infos[3],
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor_sets[i],
                    .dstBinding = 4,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &buffer_infos[4],
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor_sets[i],
                    .dstBinding = 5, // Текстура
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .pImageInfo = &image_info,
                },
                    
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor_sets[i],
                    .dstBinding = 8, // Time uniforms
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .pBufferInfo = &buffer_infos[5], // Индекс time_uniforms_buffer в массиве
                },
            };

            vkUpdateDescriptorSets(device, static_cast<uint32_t>(sizeof(write_infos) / sizeof(write_infos[0])),
                                write_infos, 0, nullptr);
        }

        // Add models to scene с указанием индекса дескрипторного набора
        // 1. Плоскость (пол) - текстура неба
        models.emplace_back(Model{
                .mesh = plane_mesh,
                .transform = Transform{},
                .albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
                .specular_color = veekay::vec3{0.5f, 0.5f, 0.5f},
                .shininess = 32.0f,
                .descriptor_set_index = 2 // Небо
        });

        // 2. Солнце (большая сфера)
        models.emplace_back(Model{
                .mesh = base_sphere_mesh,
                .transform = Transform{
                    .position = {0.0f, -0.5f, 0.0f},
                    .scale = {5.0f, 4.0f, 5.0f}
                },
                .albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f}, // Белый цвет, текстура даст цвет
                .specular_color = veekay::vec3{0.8f, 0.8f, 0.9f},
                .shininess = 128.0f,
                .descriptor_set_index = 0 // Солнце
        });

        // 3. Земля (маленькая сфера)
        models.emplace_back(Model{
                .mesh = base_sphere_mesh,
                .transform = Transform{
                    .position = {4.0f, -2.0f, 0.0f},
                    .scale = {1.0f, 1.0f, 1.0f}
                },
                .albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
                .specular_color = veekay::vec3{0.7f, 0.7f, 0.7f},
                .shininess = 64.0f,
                .descriptor_set_index = 1 // Земля
        });

        // Initialize lights
        {
            // Directional light
            DirectionalLightUBO dir_light{};
            dir_light.direction = veekay::vec3{0.0f, 1.0f, 0.0f};
            dir_light.color = veekay::vec3{1.0f, 1.0f, 0.95f};
            dir_light.intensity = 0.3f;

            if (directional_light_buffer && directional_light_buffer->mapped_region)
                *reinterpret_cast<DirectionalLightUBO*>(directional_light_buffer->mapped_region) = dir_light;

            // Spot lights in storage buffer
            std::vector<SpotLightSSBO> spot_lights(max_spot_lights);
            
            // Main spotlight
            spot_lights[0].position = veekay::vec3{0.0f, -7.857f, 0.0f};
            spot_lights[0].direction = veekay::vec3{0.0f, 1.7f, 0.0f};
            spot_lights[0].color = veekay::vec3{0.925f, 0.709f, 0.106f};
            spot_lights[0].intensity = 50.0f;

            float inner_deg = 20.0f * (3.14159265f / 180.0f);
            float outer_deg = 35.0f * (3.14159265f / 180.0f);
            spot_lights[0].inner_cos = std::cos(inner_deg);
            spot_lights[0].outer_cos = std::cos(outer_deg);

            if (spot_lights_buffer && spot_lights_buffer->mapped_region)
                memcpy(spot_lights_buffer->mapped_region, spot_lights.data(), spot_lights.size() * sizeof(SpotLightSSBO));

            // Point lights in storage buffer
            std::vector<PointLightSSBO> point_lights(max_point_lights);
            

			// First point light - красный, слева
			point_lights[0].position = veekay::vec3{-2.857f, -1.773f, 0.0f};
			point_lights[0].color = veekay::vec3{0.102f, 0.855f, 0.878f};
			point_lights[0].intensity = 25.0f;
			point_lights[0].radius = 6.091f;

			// Second point light - желтый, справа
			point_lights[1].position = veekay::vec3{3.328f, -1.547f, 0.0f};
			point_lights[1].color = veekay::vec3{0.388f, 1.000f, 0.200f}; // Желтый цвет
			point_lights[1].intensity = 44.976f;
			point_lights[1].radius = 9.727f;

			if (point_lights_buffer && point_lights_buffer->mapped_region)
				memcpy(point_lights_buffer->mapped_region, point_lights.data(), point_lights.size() * sizeof(PointLightSSBO));
        }

        std::cout << "Textures loaded successfully!\n";
    }

    void shutdown() {
        VkDevice &device = veekay::app.vk_device;

        // Уничтожаем сэмплеры
        if (sun_sampler != VK_NULL_HANDLE && sun_sampler != missing_texture_sampler) 
            vkDestroySampler(device, sun_sampler, nullptr);
        if (earth_sampler != VK_NULL_HANDLE && earth_sampler != missing_texture_sampler) 
            vkDestroySampler(device, earth_sampler, nullptr);
        if (sky_sampler != VK_NULL_HANDLE && sky_sampler != missing_texture_sampler) 
            vkDestroySampler(device, sky_sampler, nullptr);
        
        vkDestroySampler(device, missing_texture_sampler, nullptr);

        // Удаляем текстуры
        if (sun_texture != missing_texture) delete sun_texture;
        if (earth_texture != missing_texture) delete earth_texture;
        if (sky_texture != missing_texture) delete sky_texture;
        delete missing_texture;

        // Удаляем буферы мешей
        if (base_sphere_mesh.index_buffer) delete base_sphere_mesh.index_buffer;
        if (base_sphere_mesh.vertex_buffer) delete base_sphere_mesh.vertex_buffer;

        delete plane_mesh.index_buffer;
        delete plane_mesh.vertex_buffer;

        delete directional_light_buffer;
        delete spot_lights_buffer;
        delete point_lights_buffer;
        delete model_uniforms_buffer;
        delete scene_uniforms_buffer;

        vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        if (fragment_shader_module != VK_NULL_HANDLE) vkDestroyShaderModule(device, fragment_shader_module, nullptr);
        if (vertex_shader_module != VK_NULL_HANDLE) vkDestroyShaderModule(device, vertex_shader_module, nullptr);
    }

    void update(double time) {
        // Время с последнего кадра для плавного вращения
        static double last_time = 0.0;
        double delta_time = time - last_time;
        last_time = time;

        static auto start_time_point = std::chrono::steady_clock::now();
        auto current_time_point = std::chrono::steady_clock::now();
        auto elapsed = current_time_point - start_time_point;
        double current_time_seconds = std::chrono::duration<double>(elapsed).count();

        TimeUniforms time_uniforms{
            .time = static_cast<float>(current_time_seconds),
            .frequency = 2.0f,
            .amplitude = 1.0f,
            .speed = 0.2f
        };

        if (time_uniforms_buffer && time_uniforms_buffer->mapped_region) {
            *reinterpret_cast<TimeUniforms*>(time_uniforms_buffer->mapped_region) = time_uniforms;
        }

        
        // Если это первый кадр, пропускаем
        if (delta_time <= 0.0) return;
        
        // Очень медленные скорости вращения (в радианах в секунду)
        float sun_rotation_speed = 0.8f; // Солнце медленно вращается
        float earth_orbit_speed = 0.2f;   // Земля вокруг солнца
        float earth_rotation_speed = 1.2f; // Земля вокруг своей оси
        
        // Обновляем углы вращения с учетом дельты времени
        sun_rotation_angle += sun_rotation_speed * static_cast<float>(delta_time);
        earth_orbit_angle += earth_orbit_speed * static_cast<float>(delta_time);
        earth_rotation_angle += earth_rotation_speed * static_cast<float>(delta_time);
        
        // Параметры орбит
        float earth_orbit_radius = 4.0f; // Расстояние от земли до солнца
        
        // Вычисляем позиции
        // Солнце остается на месте (индекс 1)
        models[1].transform.position = veekay::vec3{0.0f, -0.5f, 0.0f};
        models[1].transform.rotation.y = sun_rotation_angle; // Вращение солнца
        
        // Земля вращается вокруг солнца
        earth_position.x = sin(earth_orbit_angle) * earth_orbit_radius;
        earth_position.z = cos(earth_orbit_angle) * earth_orbit_radius;
        earth_position.y = -2.0f; // Высота земли
        
        models[2].transform.position = earth_position;
        models[2].transform.rotation.y = earth_rotation_angle; // Вращение земли вокруг своей оси
        
        ImGui::Begin("Controls:");
        ImGui::Text("WASD/QZ move, hold left mouse to rotate.");
        ImGui::Separator();
        
        // Solar system info - ТОЛЬКО СОЛНЦЕ И ЗЕМЛЯ
        ImGui::Text("Solar System:");
        ImGui::Text("Sun rotation: %.1f°", sun_rotation_angle);
        ImGui::Text("Earth orbit: %.1f°", earth_orbit_angle);
        ImGui::Text("Earth rotation: %.1f°", earth_rotation_angle);
        
        // Позиция Земли
        ImGui::Text("Earth position: (%.2f, %.2f, %.2f)", 
                    earth_position.x, earth_position.y, earth_position.z);
        ImGui::Separator();

        // Camera controls
        ImGui::Text("Camera:");
        ImGui::Text("Position: %.2f, %.2f, %.2f", camera.position.x, camera.position.y, camera.position.z);
        ImGui::Text("Rotation: %.2f, %.2f, %.2f", camera.rotation.x, camera.rotation.y, camera.rotation.z);

        if (ImGui::Button("Reset Camera")) {
            camera.position = veekay::vec3{-0.56f, -6.85f, -8.80f};
            camera.rotation = veekay::vec3{-0.59f, -0.01f, 0.0f};
            camera.fov = Camera::default_fov;
            camera.near_plane = Camera::default_near_plane;
            camera.far_plane = Camera::default_far_plane;
        }

        ImGui::Separator();

		// Directional light controls
		ImGui::Text("Directional Light:");
		static float dir_intensity = 0.3f;
		static float dir_color[3] = {1.0f, 1.0f, 0.95f};
		ImGui::SliderFloat("Dir Intensity", &dir_intensity, 0.0f, 2.0f);
		ImGui::Separator();

		// Spotlight controls
		ImGui::Text("Spotlight Controls:");

		static float spot_pos[3] = {0.0f, -7.857f, 0.0f};
		static float spot_dir[3] = {0.0f, 1.7f, 0.0f};
		static float spot_color[3] = {0.925f, 0.709f, 0.106f};
		static float spot_intensity = 50.0f;
		static float spot_inner_angle = 20.0f;
		static float spot_outer_angle = 35.0f;

		ImGui::SliderFloat3("Position", spot_pos, -10.0f, 20.0f);
		ImGui::SliderFloat3("Direction", spot_dir, -5.0f, 5.0f);
		ImGui::ColorEdit3("Color", spot_color);
		ImGui::SliderFloat("Intensity", &spot_intensity, 0.0f, 100.0f);
		ImGui::SliderFloat("Inner Angle", &spot_inner_angle, 5.0f, 45.0f);
		ImGui::SliderFloat("Outer Angle", &spot_outer_angle, 10.0f, 60.0f);

		if (spot_inner_angle > spot_outer_angle) spot_inner_angle = spot_outer_angle;

		if (ImGui::Button("Reset Spotlight")) {
			spot_pos[0] = 0.0f;
			spot_pos[1] = -7.857f;
			spot_pos[2] = 0.0f;
			spot_dir[0] = 0.0f;
			spot_dir[1] = 1.7f;
			spot_dir[2] = 0.0f;
            spot_color[0] = 0.925f;  // #ECB51B
            spot_color[1] = 0.709f;
            spot_color[2] = 0.106f;
			spot_intensity = 50.0f;
			spot_inner_angle = 20.0f;
			spot_outer_angle = 35.0f;
		}

		ImGui::Separator();

		// Point lights controls - отдельные контролы для каждого источника
		ImGui::Text("Point Light 1 (Blue):");
		static float point1_pos[3] = {-2.857f, -1.773f, 0.0f};
		static float point1_color[3] = {0.102f, 0.855f, 0.878f};
		static float point1_intensity = 25.0f;
		static float point1_radius = 6.091f;

		ImGui::SliderFloat3("Point 1 Position", point1_pos, -10.0f, 10.0f);
		ImGui::ColorEdit3("Point 1 Color", point1_color);
		ImGui::SliderFloat("Point 1 Intensity", &point1_intensity, 0.0f, 50.0f);
		ImGui::SliderFloat("Point 1 Radius", &point1_radius, 1.0f, 20.0f);

		if (ImGui::Button("Reset Point Light 1")) {
			point1_pos[0] = -2.857f;
			point1_pos[1] = -1.773f;
			point1_pos[2] = 0.0f;
			point1_color[0] = 0.102f;
			point1_color[1] = 0.855f;
			point1_color[2] = 0.878f;
			point1_intensity = 25.0f;
			point1_radius = 6.091f;
		}

		ImGui::Separator();

		ImGui::Text("Point Light 2 (Green):");
		static float point2_pos[3] = {3.328f, -1.547f, 0.0f};
		static float point2_color[3] = {0.388f, 1.000f, 0.200f}; // Желтый вместо синего
		static float point2_intensity = 44.976f;
		static float point2_radius = 9.727f;

		ImGui::SliderFloat3("Point 2 Position", point2_pos, -10.0f, 10.0f);
		ImGui::ColorEdit3("Point 2 Color", point2_color);
		ImGui::SliderFloat("Point 2 Intensity", &point2_intensity, 0.0f, 50.0f);
		ImGui::SliderFloat("Point 2 Radius", &point2_radius, 1.0f, 20.0f);

		if (ImGui::Button("Reset Point Light 2")) {
			point2_pos[0] = 3.328f;
			point2_pos[1] = -1.547f;
			point2_pos[2] = 0.0f;
			point2_color[0] = 0.388f;
			point2_color[1] = 1.000f;
			point2_color[2] = 0.200f;
			point2_intensity = 44.976f;
			point2_radius = 9.727f;
		}

        ImGui::End();

        if (!ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {
            using namespace veekay::input;

            if (mouse::isButtonDown(mouse::Button::left)) {
                auto move_delta = mouse::cursorDelta();

                const float sensitivity = 0.005f;
                camera.rotation.y += move_delta.x * sensitivity;
                camera.rotation.x += -move_delta.y * sensitivity;

                if (camera.rotation.x > 89.0f * (3.14159265f / 180.0f)) camera.rotation.x = 89.0f * (3.14159265f / 180.0f);
                if (camera.rotation.x < -89.0f * (3.14159265f / 180.0f)) camera.rotation.x = -89.0f * (3.14159265f / 180.0f);

                float yaw = camera.rotation.y;
                float pitch = camera.rotation.x;

                veekay::vec3 front;
                front.x = sin(yaw) * cos(pitch);
                front.y = -sin(pitch);
                front.z = cos(yaw) * cos(pitch);
                front = veekay::vec3::normalized(front);

                veekay::vec3 world_up = {0.0f, -1.0f, 0.0f};
                veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(world_up, front));
                veekay::vec3 up = veekay::vec3::cross(front, right);

                const float move_speed = 0.1f;

                if (keyboard::isKeyDown(keyboard::Key::w))
                    camera.position += front * move_speed;
                if (keyboard::isKeyDown(keyboard::Key::s))
                    camera.position -= front * move_speed;
                if (keyboard::isKeyDown(keyboard::Key::d))
                    camera.position -= right * move_speed;
                if (keyboard::isKeyDown(keyboard::Key::a))
                    camera.position += right * move_speed;
                if (keyboard::isKeyDown(keyboard::Key::q))
                    camera.position += up * move_speed;
                if (keyboard::isKeyDown(keyboard::Key::z))
                    camera.position -= up * move_speed;
            }
        }

        float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
        SceneUniforms scene_uniforms{
                .view_projection = camera.view_projection(aspect_ratio),
                .camera_position = camera.position,
                .spot_light_count = spot_light_count,
                .point_light_count = point_light_count,
        };

        std::vector<ModelUniforms> model_uniforms(models.size());
        for (size_t i = 0, n = models.size(); i < n; ++i) {
            const Model &model = models[i];
            ModelUniforms &uniforms = model_uniforms[i];
            uniforms.model = model.transform.matrix();
            uniforms.albedo_color = model.albedo_color;
            uniforms.specular_color = model.specular_color;
            uniforms.shininess = model.shininess;
        }

        // Copy scene uniforms
        if (scene_uniforms_buffer && scene_uniforms_buffer->mapped_region)
            *(SceneUniforms *) scene_uniforms_buffer->mapped_region = scene_uniforms;

        // Copy model uniforms
        const size_t alignment = veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));
        for (size_t i = 0, n = model_uniforms.size(); i < n; ++i) {
            const ModelUniforms &uniforms = model_uniforms[i];
            char *const pointer = static_cast<char *>(model_uniforms_buffer->mapped_region) + i * alignment;
            *reinterpret_cast<ModelUniforms *>(pointer) = uniforms;
        }

        // Update directional light
        if (directional_light_buffer && directional_light_buffer->mapped_region) {
            DirectionalLightUBO* dir_light = reinterpret_cast<DirectionalLightUBO*>(directional_light_buffer->mapped_region);
            
            float rot = static_cast<float>(time) * 0.25f;
            dir_light->direction = veekay::vec3{std::sin(rot), -1.0f, std::cos(rot)};
            dir_light->direction = veekay::vec3::normalized(dir_light->direction);
            dir_light->color = veekay::vec3{dir_color[0], dir_color[1], dir_color[2]};
            dir_light->intensity = dir_intensity;
        }

        // Update spot lights in storage buffer
        if (spot_lights_buffer && spot_lights_buffer->mapped_region) {
            SpotLightSSBO* spot_lights = reinterpret_cast<SpotLightSSBO*>(spot_lights_buffer->mapped_region);
            
            // Update first spotlight
            spot_lights[0].position = veekay::vec3{spot_pos[0], spot_pos[1], spot_pos[2]};
            spot_lights[0].direction = veekay::vec3::normalized(veekay::vec3{spot_dir[0], spot_dir[1], spot_dir[2]});
            spot_lights[0].color = veekay::vec3{spot_color[0], spot_color[1], spot_color[2]};
            spot_lights[0].intensity = spot_intensity;

            float inner_rad = spot_inner_angle * (3.14159265f / 180.0f);
            float outer_rad = spot_outer_angle * (3.14159265f / 180.0f);
            spot_lights[0].inner_cos = std::cos(inner_rad);
            spot_lights[0].outer_cos = std::cos(outer_rad);
        }

        // Update point lights in storage buffer
        if (point_lights_buffer && point_lights_buffer->mapped_region) {
            PointLightSSBO* point_lights = reinterpret_cast<PointLightSSBO*>(point_lights_buffer->mapped_region);
            
            // Update first point light (красный, слева)
            point_lights[0].position = veekay::vec3{point1_pos[0], point1_pos[1], point1_pos[2]};
            point_lights[0].color = veekay::vec3{point1_color[0], point1_color[1], point1_color[2]};
            point_lights[0].intensity = point1_intensity;
            point_lights[0].radius = point1_radius;

            // Update second point light (желтый, справа)
            point_lights[1].position = veekay::vec3{point2_pos[0], point2_pos[1], point2_pos[2]};
            point_lights[1].color = veekay::vec3{point2_color[0], point2_color[1], point2_color[2]};
            point_lights[1].intensity = point2_intensity;
            point_lights[1].radius = point2_radius;
        }
    }

    void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
        vkResetCommandBuffer(cmd, 0);

        {
            VkCommandBufferBeginInfo info{
                    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };
            vkBeginCommandBuffer(cmd, &info);
        }

        {
            VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
            VkClearValue clear_depth{.depthStencil = {1.0f, 0}};
            VkClearValue clear_values[] = {clear_color, clear_depth};

            VkRenderPassBeginInfo info{
                    .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                    .renderPass = veekay::app.vk_render_pass,
                    .framebuffer = framebuffer,
                    .renderArea = {
                            .extent = {veekay::app.window_width, veekay::app.window_height},
                    },
                    .clearValueCount = 2,
                    .pClearValues = clear_values,
            };

            vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        VkDeviceSize zero_offset = 0;

        VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
        VkBuffer current_index_buffer = VK_NULL_HANDLE;

        const size_t model_uniforms_alignment =
                veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

        for (size_t i = 0, n = models.size(); i < n; ++i) {
            const Model &model = models[i];
            const Mesh &mesh = model.mesh;

            if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
                current_vertex_buffer = mesh.vertex_buffer->buffer;
                vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
            }

            if (current_index_buffer != mesh.index_buffer->buffer) {
                current_index_buffer = mesh.index_buffer->buffer;
                vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
            }

            // dynamic offset for per-model UBO
            uint32_t offset = static_cast<uint32_t>(i * model_uniforms_alignment);
            
            // Используем соответствующий дескрипторный набор для этой модели
            VkDescriptorSet set_to_bind = descriptor_sets[model.descriptor_set_index];
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                    0, 1, &set_to_bind, 1, &offset);

            vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
        }

        vkCmdEndRenderPass(cmd);
        vkEndCommandBuffer(cmd);
    }

} // namespace

int main() {
    return veekay::run({
                               .init = initialize,
                               .shutdown = shutdown,
                               .update = update,
                               .render = render,
                       });
}