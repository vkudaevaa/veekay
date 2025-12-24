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
    constexpr uint32_t shadow_map_size = 2048; // разрешение карты теней

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
        veekay::mat4 dir_light_matrix; // матрицы проекции теней
        veekay::mat4 spot_light_matrix;
        veekay::vec3 camera_position;
        float _pad0;
        uint32_t spot_light_count;
        float _pad1[3];
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

    struct ShadowPushConstant {
        veekay::mat4 light_view_proj; // матрица вида и проекции для текущего света
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
        uint32_t descriptor_set_index; // индекс в массиве дескрипторных наборов
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
        
        float sun_rotation_angle = 0.0f;
        float earth_orbit_angle = 0.0f;
        float earth_rotation_angle = 0.0f;
        
        veekay::vec3 earth_position;
    }

    inline namespace {
        // инициализация шейдерных модулей
        VkShaderModule vertex_shader_module = VK_NULL_HANDLE;
        VkShaderModule fragment_shader_module = VK_NULL_HANDLE;
        VkShaderModule shadow_vertex_shader_module = VK_NULL_HANDLE;

        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout shadow_descriptor_set_layout = VK_NULL_HANDLE;

        std::vector<VkDescriptorSet> descriptor_sets; // Множество дескрипторных наборов
        VkDescriptorSet shadow_descriptor_set = VK_NULL_HANDLE;

        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;

        // как будет происходить рендеринг глубины для создания карты теней, 
        // независимо от того, для какого источника света она создается.
        VkPipelineLayout shadow_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline shadow_pipeline = VK_NULL_HANDLE; // конвейер для теней
        VkRenderPass shadow_render_pass = VK_NULL_HANDLE;
        VkSampler shadow_sampler = VK_NULL_HANDLE;

        // карта теней для направленного света
        VkFramebuffer shadow_dir_framebuffer = VK_NULL_HANDLE;
        VkImage shadow_dir_image = VK_NULL_HANDLE;
        VkDeviceMemory shadow_dir_memory = VK_NULL_HANDLE;
        VkImageView shadow_dir_view = VK_NULL_HANDLE;

        // для прожекторного
        VkFramebuffer shadow_spot_framebuffer = VK_NULL_HANDLE;
        VkImage shadow_spot_image = VK_NULL_HANDLE;
        VkDeviceMemory shadow_spot_memory = VK_NULL_HANDLE;
        VkImageView shadow_spot_view = VK_NULL_HANDLE;

        veekay::graphics::Buffer *scene_uniforms_buffer = nullptr;
        veekay::graphics::Buffer *model_uniforms_buffer = nullptr;
        veekay::graphics::Buffer *directional_light_buffer = nullptr;
        veekay::graphics::Buffer *spot_lights_buffer = nullptr;
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

    veekay::mat4 mat4_ortho(float left, float right, float bottom, float top, float zNear, float zFar) {
        veekay::mat4 res = veekay::mat4::identity();
        res.columns[0].x = 2.0f / (right - left);
        res.columns[1].y = 2.0f / (bottom - top);
        res.columns[2].z = 1.0f / (zFar - zNear);
        res.columns[3].x = -(right + left) / (right - left);
        res.columns[3].y = -(top + bottom) / (bottom - top);
        res.columns[3].z = -zNear / (zFar - zNear);
        return res;
    }

    veekay::mat4 mat4_lookAt(veekay::vec3 eye, veekay::vec3 center, veekay::vec3 up) {
        veekay::vec3 f = veekay::vec3::normalized(center - eye);
        veekay::vec3 s = veekay::vec3::normalized(veekay::vec3::cross(f, up));
        veekay::vec3 u = veekay::vec3::cross(s, f);
        veekay::mat4 res = veekay::mat4::identity();
        res.columns[0].x = s.x;
        res.columns[1].x = s.y;
        res.columns[2].x = s.z;
        res.columns[0].y = u.x;
        res.columns[1].y = u.y;
        res.columns[2].y = u.z;
        res.columns[0].z = f.x;
        res.columns[1].z = f.y;
        res.columns[2].z = f.z;
        res.columns[3].x = -veekay::vec3::dot(s, eye);
        res.columns[3].y = -veekay::vec3::dot(u, eye);
        res.columns[3].z = -veekay::vec3::dot(f, eye);
        return res;
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

    Mesh createSphereMesh(float radius = 0.5f, uint32_t segments = 64) {
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

    uint32_t findMemoryType(uint32_t typeFilter) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(veekay::app.vk_physical_device, &memProperties);
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) &&
                (memProperties.memoryTypes[i].propertyFlags & 1) == 1)
                return i;
        }

        return 0;
    }

    void createShadowMapResource(VkDevice device, VkImage &image, VkDeviceMemory &mem, VkImageView &view) {
        VkImageCreateInfo imageInfo{
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .imageType = VK_IMAGE_TYPE_2D,
                .format = VK_FORMAT_D32_SFLOAT, // формат для хранения глубины
                .extent = {shadow_map_size, shadow_map_size, 1},
                .mipLevels = 1, 
                .arrayLayers = 1, 
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE, 
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
        };
        // создание физического изображения
        vkCreateImage(device, &imageInfo, nullptr, &image);

        VkMemoryRequirements memReq;
        vkGetImageMemoryRequirements(device, image, &memReq);
        VkMemoryAllocateInfo allocInfo{
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .allocationSize = memReq.size,
                .memoryTypeIndex = findMemoryType(memReq.memoryTypeBits)
        };
        vkAllocateMemory(device, &allocInfo, nullptr, &mem);
        // привязываем изображение к физической памяти
        vkBindImageMemory(device, image, mem, 0); 

        VkImageViewCreateInfo viewInfo{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                // привязка к созданному физическому изображению
                .image = image, 
                .viewType = VK_IMAGE_VIEW_TYPE_2D, 
                .format = VK_FORMAT_D32_SFLOAT,
                .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1}
        };
        // создаём представление изображения
        vkCreateImageView(device, &viewInfo, nullptr, &view);
    }

    void initialize(VkCommandBuffer cmd) {
        VkDevice &device = veekay::app.vk_device;
        VkPhysicalDevice &physical_device = veekay::app.vk_physical_device;

        {   
            // создаём сэмплер теней
            VkSamplerCreateInfo samplerInfo{
                    .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                    .magFilter = VK_FILTER_LINEAR, 
                    .minFilter = VK_FILTER_LINEAR,
                    .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                    .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                    .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                    .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                    .mipLodBias = 0.0f,
                    .compareEnable = VK_TRUE, 
                    .compareOp = VK_COMPARE_OP_LESS,
                    .minLod = 0.0f, 
                    .maxLod = 1.0f, 
                    .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE
            };
            vkCreateSampler(device, &samplerInfo, nullptr, &shadow_sampler);

            // структура, куда записываются свойства карты теней
            VkAttachmentDescription depthAttachment{
                    .format = VK_FORMAT_D32_SFLOAT, 
                    .samples = VK_SAMPLE_COUNT_1_BIT,
                    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, 
                    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                    .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, 
                    .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, 
                    .finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            };

            VkAttachmentReference depthRef{0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
            VkSubpassDescription subpass{
                .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS, 
                .pDepthStencilAttachment = &depthRef
            };
            
            VkSubpassDependency dependencies[2]; // определяет барьеры синхронизации между проходами
            dependencies[0] = {
                VK_SUBPASS_EXTERNAL, 0, 
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 
                VK_ACCESS_SHADER_READ_BIT,
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, 
                VK_DEPENDENCY_BY_REGION_BIT
            };
            dependencies[1] = {
                0, VK_SUBPASS_EXTERNAL, 
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT, 
                VK_DEPENDENCY_BY_REGION_BIT
            };
            
            VkRenderPassCreateInfo rpInfo{
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, 
                .attachmentCount = 1, 
                .pAttachments = &depthAttachment, 
                .subpassCount = 1, 
                .pSubpasses = &subpass, 
                .dependencyCount = 2, 
                .pDependencies = dependencies
            };
            vkCreateRenderPass(device, &rpInfo, nullptr, &shadow_render_pass);

            // создание карты теней и фреймбуфера
            createShadowMapResource(device, shadow_dir_image, shadow_dir_memory, shadow_dir_view);
            VkFramebufferCreateInfo fbDirInfo{
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, 
                .renderPass = shadow_render_pass, 
                .attachmentCount = 1, 
                .pAttachments = &shadow_dir_view, 
                .width = shadow_map_size, 
                .height = shadow_map_size, 
                .layers = 1
            };
            vkCreateFramebuffer(device, &fbDirInfo, nullptr, &shadow_dir_framebuffer);

            createShadowMapResource(device, shadow_spot_image, shadow_spot_memory, shadow_spot_view);
            VkFramebufferCreateInfo fbSpotInfo{
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, 
                .renderPass = shadow_render_pass, 
                .attachmentCount = 1, 
                .pAttachments = &shadow_spot_view, 
                .width = shadow_map_size, 
                .height = shadow_map_size, 
                .layers = 1
            };
            vkCreateFramebuffer(device, &fbSpotInfo, nullptr, &shadow_spot_framebuffer);
        }

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

            shadow_vertex_shader_module = loadShaderModule("./shaders/shadow.vert.spv");
            if (shadow_vertex_shader_module == VK_NULL_HANDLE) {
                std::cerr << "Failed to load Vulkan shadow vertex shader from file\n";
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

            VkVertexInputAttributeDescription shadow_attributes[] = {
                {
                    .location = 0,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(Vertex, position),
                },
            };

            VkPipelineVertexInputStateCreateInfo shadow_input_state_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                    .vertexBindingDescriptionCount = 1,
                    .pVertexBindingDescriptions = &buffer_binding,
                    .vertexAttributeDescriptionCount = 1,
                    .pVertexAttributeDescriptions = shadow_attributes,
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

            VkPipelineRasterizationStateCreateInfo shadow_raster_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                    .polygonMode = VK_POLYGON_MODE_FILL,
                    .cullMode = VK_CULL_MODE_BACK_BIT,
                    .frontFace = VK_FRONT_FACE_CLOCKWISE,
                    .depthBiasEnable = VK_TRUE,
                    .depthBiasConstantFactor = 4.5f,
                    .depthBiasSlopeFactor = 4.5f,
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

            VkViewport shadow_viewport{
                    .x = 0.0f,
                    .y = 0.0f,
                    .width = static_cast<float>(shadow_map_size),
                    .height = static_cast<float>(shadow_map_size),
                    .minDepth = 0.0f,
                    .maxDepth = 1.0f,
            };

            VkRect2D shadow_scissor{
                    .offset = {0, 0},
                    .extent = {shadow_map_size, shadow_map_size},
            };

            VkPipelineViewportStateCreateInfo shadow_viewport_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                    .viewportCount = 1,
                    .pViewports = &shadow_viewport,
                    .scissorCount = 1,
                    .pScissors = &shadow_scissor,
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
                        .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        .descriptorCount = 16,
                    },
                    {
                        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        .descriptorCount = 8,
                    }
                };

                VkDescriptorPoolCreateInfo info{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                        .maxSets = 8,
                        .poolSizeCount = static_cast<uint32_t>(sizeof(pools) / sizeof(pools[0])),
                        .pPoolSizes = pools,
                };

                if (vkCreateDescriptorPool(device, &info, nullptr, &descriptor_pool) != VK_SUCCESS) {
                    std::cerr << "Failed to create Vulkan descriptor pool\n";
                    veekay::app.running = false;
                    return;
                }
            }

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
                        .binding = 5,
                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                    },
                    {
                        .binding = 8,
                        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                    },
                    {
                        .binding = 9,
                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                    },
                    {
                        .binding = 10,
                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
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

            // Shadow descriptor set layout
            {
                VkDescriptorSetLayoutBinding bindings[] = {
                    {
                        .binding = 0,
                        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
                    },
                    {
                        .binding = 1,
                        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
                    },
                };

                VkDescriptorSetLayoutCreateInfo info{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                        .bindingCount = static_cast<uint32_t>(sizeof(bindings) / sizeof(bindings[0])),
                        .pBindings = bindings,
                };

                if (vkCreateDescriptorSetLayout(device, &info, nullptr, &shadow_descriptor_set_layout) != VK_SUCCESS) {
                    std::cerr << "Failed to create Vulkan shadow descriptor set layout\n";
                    veekay::app.running = false;
                    return;
                }
            }

            // Pipeline layouts
            VkPushConstantRange pc_range{
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
                .offset = 0,
                .size = sizeof(ShadowPushConstant)
            };

            VkPipelineLayoutCreateInfo shadow_layout_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                    .setLayoutCount = 1,
                    .pSetLayouts = &shadow_descriptor_set_layout,
                    .pushConstantRangeCount = 1,
                    .pPushConstantRanges = &pc_range,
            };

            if (vkCreatePipelineLayout(device, &shadow_layout_info, nullptr, &shadow_pipeline_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan shadow pipeline layout\n";
                veekay::app.running = false;
                return;
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

            // Shadow pipeline
            {
                VkPipelineShaderStageCreateInfo shadow_stage{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_VERTEX_BIT,
                    .module = shadow_vertex_shader_module,
                    .pName = "main",
                };

                VkPipelineDepthStencilStateCreateInfo shadow_depth_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                    .depthTestEnable = true,
                    .depthWriteEnable = true,
                    .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
                };

                VkGraphicsPipelineCreateInfo info{
                    .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                    .stageCount = 1,
                    .pStages = &shadow_stage,
                    .pVertexInputState = &shadow_input_state_info,
                    .pInputAssemblyState = &assembly_state_info,
                    .pViewportState = &shadow_viewport_info,
                    .pRasterizationState = &shadow_raster_info,
                    .pMultisampleState = &sample_info,
                    .pDepthStencilState = &shadow_depth_info,
                    .layout = shadow_pipeline_layout,
                    .renderPass = shadow_render_pass,
                };

                if (vkCreateGraphicsPipelines(device, nullptr, 1, &info, nullptr, &shadow_pipeline) != VK_SUCCESS) {
                    std::cerr << "Failed to create Vulkan shadow pipeline\n";
                    veekay::app.running = false;
                    return;
                }
            }

            // Main pipeline
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
                {{-plane_size, 0.0f, plane_size},  {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
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

        // Создаем дескрипторные наборы для разных материалов (с тенями)
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
                    .buffer = time_uniforms_buffer->buffer,
                    .offset = 0,
                    .range = sizeof(TimeUniforms),
                },
            };

            VkDescriptorImageInfo image_info{
                .sampler = sampler_to_use,
                .imageView = texture_to_use->view,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };

            VkDescriptorImageInfo dir_shadow_image_info{
                .sampler = shadow_sampler,
                .imageView = shadow_dir_view,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };

            VkDescriptorImageInfo spot_shadow_image_info{
                .sampler = shadow_sampler,
                .imageView = shadow_spot_view,
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
                    .dstBinding = 5,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .pImageInfo = &image_info,
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor_sets[i],
                    .dstBinding = 8,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .pBufferInfo = &buffer_infos[4],
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor_sets[i],
                    .dstBinding = 9,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .pImageInfo = &dir_shadow_image_info,
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor_sets[i],
                    .dstBinding = 10,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .pImageInfo = &spot_shadow_image_info,
                },
            };

            vkUpdateDescriptorSets(device, static_cast<uint32_t>(sizeof(write_infos) / sizeof(write_infos[0])),
                                write_infos, 0, nullptr);
        }

        // Allocate shadow descriptor set
        {
            VkDescriptorSetAllocateInfo alloc_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = descriptor_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &shadow_descriptor_set_layout,
            };

            if (vkAllocateDescriptorSets(device, &alloc_info, &shadow_descriptor_set) != VK_SUCCESS) {
                std::cerr << "Failed to allocate shadow descriptor set\n";
                veekay::app.running = false;
                return;
            }

            VkDescriptorBufferInfo buffer_info{
                .buffer = model_uniforms_buffer->buffer,
                .offset = 0,
                .range = sizeof(ModelUniforms),
            };

            VkWriteDescriptorSet write_info{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = shadow_descriptor_set,
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                .pBufferInfo = &buffer_info,
            };

            vkUpdateDescriptorSets(device, 1, &write_info, 0, nullptr);
        }

        models.emplace_back(Model{
                .mesh = plane_mesh,
                .transform = Transform{},
                .albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
                .specular_color = veekay::vec3{0.5f, 0.5f, 0.5f},
                .shininess = 32.0f,
                .descriptor_set_index = 2 // Небо
        });

        models.emplace_back(Model{
                .mesh = base_sphere_mesh,
                .transform = Transform{
                    .position = {0.0f, -0.5f, 0.0f},
                    .scale = {5.0f, 4.0f, 5.0f}
                },
                .albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
                .specular_color = veekay::vec3{0.8f, 0.8f, 0.9f},
                .shininess = 128.0f,
                .descriptor_set_index = 0 // Солнце
        });

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
            DirectionalLightUBO dir_light{};
            dir_light.direction = veekay::vec3{0.0f, 1.0f, 0.0f};
            dir_light.color = veekay::vec3{1.0f, 1.0f, 0.95f};
            dir_light.intensity = 0.3f;

            if (directional_light_buffer && directional_light_buffer->mapped_region)
                *reinterpret_cast<DirectionalLightUBO*>(directional_light_buffer->mapped_region) = dir_light;

            std::vector<SpotLightSSBO> spot_lights(max_spot_lights);

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
        }

        std::cout << "Textures loaded successfully!\n";
    }

	void shutdown() {
		VkDevice &device = veekay::app.vk_device;

        auto cleanupShadow = [&](VkFramebuffer fb, VkImageView view, VkImage img, VkDeviceMemory mem) {
            vkDestroyFramebuffer(device, fb, nullptr);
            vkDestroyImageView(device, view, nullptr);
            vkDestroyImage(device, img, nullptr);
            vkFreeMemory(device, mem, nullptr);
        };

        cleanupShadow(shadow_dir_framebuffer, shadow_dir_view, shadow_dir_image, shadow_dir_memory);
        cleanupShadow(shadow_spot_framebuffer, shadow_spot_view, shadow_spot_image, shadow_spot_memory);

        vkDestroyRenderPass(device, shadow_render_pass, nullptr);
        vkDestroySampler(device, shadow_sampler, nullptr);
        vkDestroyPipeline(device, shadow_pipeline, nullptr);
        vkDestroyPipelineLayout(device, shadow_pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(device, shadow_descriptor_set_layout, nullptr);

        if (shadow_vertex_shader_module != VK_NULL_HANDLE) 
            vkDestroyShaderModule(device, shadow_vertex_shader_module, nullptr);

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

		// Удаляем все буферы
		delete directional_light_buffer;
		delete spot_lights_buffer;
		delete model_uniforms_buffer;
		delete scene_uniforms_buffer;
		delete time_uniforms_buffer;

		vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
		vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

		vkDestroyPipeline(device, pipeline, nullptr);
		vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
		if (fragment_shader_module != VK_NULL_HANDLE) vkDestroyShaderModule(device, fragment_shader_module, nullptr);
		if (vertex_shader_module != VK_NULL_HANDLE) vkDestroyShaderModule(device, vertex_shader_module, nullptr);
	}

    void update(double time) {
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

        if (delta_time <= 0.0) return;

        float sun_rotation_speed = 0.8f;
        float earth_orbit_speed = 0.2f;
        float earth_rotation_speed = 1.2f;
        
        sun_rotation_angle += sun_rotation_speed * static_cast<float>(delta_time);
        earth_orbit_angle += earth_orbit_speed * static_cast<float>(delta_time);
        earth_rotation_angle += earth_rotation_speed * static_cast<float>(delta_time);
        
        float earth_orbit_radius = 4.0f;

        models[1].transform.position = veekay::vec3{0.0f, -0.5f, 0.0f};
        models[1].transform.rotation.y = sun_rotation_angle;
        
        earth_position.x = sin(earth_orbit_angle) * earth_orbit_radius;
        earth_position.z = cos(earth_orbit_angle) * earth_orbit_radius;
        earth_position.y = -2.0f;
        
        models[2].transform.position = earth_position;
        models[2].transform.rotation.y = earth_rotation_angle;
        
        ImGui::Begin("Controls:");
        ImGui::Text("WASD/QZ move, hold left mouse to rotate.");
        ImGui::Separator();

        ImGui::Text("Solar System:");
        ImGui::Text("Sun rotation: %.1f°", sun_rotation_angle);
        ImGui::Text("Earth orbit: %.1f°", earth_orbit_angle);
        ImGui::Text("Earth rotation: %.1f°", earth_rotation_angle);

        ImGui::Text("Earth position: (%.2f, %.2f, %.2f)", 
                    earth_position.x, earth_position.y, earth_position.z);
        ImGui::Separator();

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

		ImGui::Text("Directional Light:");
		static float dir_intensity = 0.3f;
		static float dir_color[3] = {1.0f, 1.0f, 0.95f};
		ImGui::SliderFloat("Dir Intensity", &dir_intensity, 0.0f, 2.0f);
		ImGui::Separator();

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
            spot_color[0] = 0.925f;
            spot_color[1] = 0.709f;
            spot_color[2] = 0.106f;
			spot_intensity = 50.0f;
			spot_inner_angle = 20.0f;
			spot_outer_angle = 35.0f;
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

        // Calculate light matrices
        // 1. Directional Light Matrix
        veekay::vec3 dirL = {0, -1, 0};
        if (directional_light_buffer && directional_light_buffer->mapped_region)
            dirL = reinterpret_cast<DirectionalLightUBO*>(directional_light_buffer->mapped_region)->direction;

        veekay::vec3 lightOrigin = {0, 0, 0};
        veekay::vec3 lightUp = {0, 1, 0};
        veekay::vec3 lightPos = lightOrigin - dirL * 10.0f;

        veekay::mat4 dirView = mat4_lookAt(lightPos, lightOrigin, lightUp);
        veekay::mat4 dirProj = mat4_ortho(-20, 20, -20, 20, 0.1f, 100.0f);
        veekay::mat4 dirMatrix = dirView * dirProj;

        // 2. Spot Light Matrix
        veekay::vec3 spotPos = {spot_pos[0], spot_pos[1], spot_pos[2]};
        veekay::vec3 spotDir = veekay::vec3::normalized({spot_dir[0], spot_dir[1], spot_dir[2]});
        veekay::vec3 spotUP = {0, 1, 0};

        if (std::abs(spotDir.y) > 0.99f) spotUP = {1, 0, 0};

        veekay::mat4 spotView = mat4_lookAt(spotPos, spotPos + spotDir, spotUP);
        veekay::mat4 spotProj = veekay::mat4::projection(spot_outer_angle * 2.0f, 1.0f, 0.1f, 50.0f);
        veekay::mat4 spotMatrix = spotView * spotProj;

        float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
        SceneUniforms scene_uniforms{
                .view_projection = camera.view_projection(aspect_ratio),
                .dir_light_matrix = dirMatrix,
                .spot_light_matrix = spotMatrix,
                .camera_position = camera.position,
                .spot_light_count = spot_light_count,
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

        if (scene_uniforms_buffer && scene_uniforms_buffer->mapped_region)
            *(SceneUniforms *) scene_uniforms_buffer->mapped_region = scene_uniforms;

        const size_t alignment = veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));
        for (size_t i = 0, n = model_uniforms.size(); i < n; ++i) {
            const ModelUniforms &uniforms = model_uniforms[i];
            char *const pointer = static_cast<char *>(model_uniforms_buffer->mapped_region) + i * alignment;
            *reinterpret_cast<ModelUniforms *>(pointer) = uniforms;
        }


        if (directional_light_buffer && directional_light_buffer->mapped_region) {
            DirectionalLightUBO* dir_light = reinterpret_cast<DirectionalLightUBO*>(directional_light_buffer->mapped_region);
            
            dir_light->direction = veekay::vec3{0.5f, 1.0f, 0.2f}; 
            dir_light->direction = veekay::vec3::normalized(dir_light->direction);
            dir_light->color = veekay::vec3{dir_color[0], dir_color[1], dir_color[2]};
            dir_light->intensity = dir_intensity;
        }

        if (spot_lights_buffer && spot_lights_buffer->mapped_region) {
            SpotLightSSBO* spot_lights = reinterpret_cast<SpotLightSSBO*>(spot_lights_buffer->mapped_region);
            
            spot_lights[0].position = veekay::vec3{spot_pos[0], spot_pos[1], spot_pos[2]};
            spot_lights[0].direction = veekay::vec3::normalized(veekay::vec3{spot_dir[0], spot_dir[1], spot_dir[2]});
            spot_lights[0].color = veekay::vec3{spot_color[0], spot_color[1], spot_color[2]};
            spot_lights[0].intensity = spot_intensity;

            float inner_rad = spot_inner_angle * (3.14159265f / 180.0f);
            float outer_rad = spot_outer_angle * (3.14159265f / 180.0f);
            spot_lights[0].inner_cos = std::cos(inner_rad);
            spot_lights[0].outer_cos = std::cos(outer_rad);
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

        size_t align = veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));
        VkDeviceSize zero_offset = 0;

        auto renderShadowPass = [&](VkFramebuffer fb, const veekay::mat4 &lightMatrix) {
            VkClearValue clear_val{.depthStencil = {1.0f, 0}};
            VkRenderPassBeginInfo rp_info{
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                .renderPass = shadow_render_pass,
                .framebuffer = fb,
                .renderArea = {{0, 0}, {shadow_map_size, shadow_map_size}},
                .clearValueCount = 1,
                .pClearValues = &clear_val
            };

            vkCmdBeginRenderPass(cmd, &rp_info, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline);
            
            ShadowPushConstant push_constants{lightMatrix};
            vkCmdPushConstants(cmd, shadow_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 
                              0, sizeof(ShadowPushConstant), &push_constants);

            for (size_t i = 0; i < models.size(); ++i) {
                vkCmdBindVertexBuffers(cmd, 0, 1, &models[i].mesh.vertex_buffer->buffer, &zero_offset);
                vkCmdBindIndexBuffer(cmd, models[i].mesh.index_buffer->buffer, 0, VK_INDEX_TYPE_UINT32);
                uint32_t dyn_offset = static_cast<uint32_t>(i * align);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_layout, 
                                        0, 1, &shadow_descriptor_set, 1, &dyn_offset);
                vkCmdDrawIndexed(cmd, models[i].mesh.indices, 1, 0, 0, 0);
            }
            vkCmdEndRenderPass(cmd);
        };

        SceneUniforms *scene_uni = (SceneUniforms *) scene_uniforms_buffer->mapped_region;

        renderShadowPass(shadow_dir_framebuffer, scene_uni->dir_light_matrix);
        renderShadowPass(shadow_spot_framebuffer, scene_uni->spot_light_matrix);

        // Main render pass
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
        VkDeviceSize vertex_offset = 0;

        VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
        VkBuffer current_index_buffer = VK_NULL_HANDLE;

        for (size_t i = 0, n = models.size(); i < n; ++i) {
            const Model &model = models[i];
            const Mesh &mesh = model.mesh;

            if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
                current_vertex_buffer = mesh.vertex_buffer->buffer;
                vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &vertex_offset);
            }

            if (current_index_buffer != mesh.index_buffer->buffer) {
                current_index_buffer = mesh.index_buffer->buffer;
                vkCmdBindIndexBuffer(cmd, current_index_buffer, 0, VK_INDEX_TYPE_UINT32);
            }

            uint32_t offset = static_cast<uint32_t>(i * align);
            
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