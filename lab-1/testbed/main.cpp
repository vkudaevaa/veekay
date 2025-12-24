#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#include <veekay/veekay.hpp>

#include <imgui.h>
#include <vulkan/vulkan_core.h>

namespace {

    constexpr float camera_fov = 70.0f;
    constexpr float camera_near_plane = 0.01f;
    constexpr float camera_far_plane = 100.0f;

    struct Matrix {
        float m[4][4];
    };

    struct Vector {
        float x, y, z;
    };

    struct Color {
        float r, g, b;
    };

    struct Vertex {
        Vector position;
        Color color;
    };

    struct ShaderConstants {
        Matrix projection;
        Matrix transform;
    };

    struct VulkanBuffer {
        VkBuffer buffer;
        VkDeviceMemory memory;
    };

    VkShaderModule vertex_shader_module = VK_NULL_HANDLE;
    VkShaderModule fragment_shader_module = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    VulkanBuffer vertex_buffer{};
    VulkanBuffer index_buffer{};
    uint32_t g_index_count = 0;

    Vector model_position = {0.0f, 0.0f, 5.0f}; 
    float model_rotation = 0.0f;
    bool model_spin = true;

    // Параметры пульсации/анимации
    float pulse_scale = 1.0f;
    float animation_time = 0.0f;
    float last_update_time = 0.0f;
    bool animation_paused = false;
    bool animation_reversed = false;
    float animation_speed = 1.0f;

    float camera_yaw = 0.0f; // угол поворота камеры вокруг верстикальной оси
    float camera_pitch = 0.0f; // вокруг горизонтальной

    bool use_perspective = true;


    Matrix identity() {
        Matrix r{};
        r.m[0][0] = r.m[1][1] = r.m[2][2] = r.m[3][3] = 1.0f;
        return r;
    }

    Matrix multiply(const Matrix& a, const Matrix& b) {
        Matrix r{};
        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < 4; ++i) {
                float s = 0.0f;
                for (int k = 0; k < 4; ++k) s += a.m[j][k] * b.m[k][i];
                r.m[j][i] = s;
            }
        }
        return r;
    }

    Matrix translation(Vector v) {
        Matrix r = identity();
        r.m[3][0] = v.x;
        r.m[3][1] = v.y;
        r.m[3][2] = v.z;
        return r;
    }

    Matrix scale(Vector s) {
        Matrix r{};
        r.m[0][0] = s.x;
        r.m[1][1] = s.y;
        r.m[2][2] = s.z;
        r.m[3][3] = 1.0f;
        return r;
    }

    Matrix rotation_axis_angle(Vector axis, float angle) {
        Matrix r{};
        float len = sqrtf(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
        if (len == 0.0f) return identity();
        axis.x /= len; axis.y /= len; axis.z /= len;

        float s = sinf(angle);
        float c = cosf(angle);
        float v = 1.0f - c;

        r.m[0][0] = axis.x * axis.x * v + c;
        r.m[0][1] = axis.x * axis.y * v + axis.z * s;
        r.m[0][2] = axis.x * axis.z * v - axis.y * s;

        r.m[1][0] = axis.y * axis.x * v - axis.z * s;
        r.m[1][1] = axis.y * axis.y * v + c;
        r.m[1][2] = axis.y * axis.z * v + axis.x * s;

        r.m[2][0] = axis.z * axis.x * v + axis.y * s;
        r.m[2][1] = axis.z * axis.y * v - axis.x * s;
        r.m[2][2] = axis.z * axis.z * v + c;

        r.m[3][3] = 1.0f;
        return r;
    }

    Matrix rotation_x(float a) {
        return rotation_axis_angle({1.0f, 0.0f, 0.0f}, a);
    }

    Matrix rotation_y(float a) {
        return rotation_axis_angle({0.0f, 1.0f, 0.0f}, a);
    }

    Matrix projection(float fov_deg, float aspect, float near, float far) {
        Matrix r{};
        float radians = fov_deg * M_PI / 180.0f;
        float f = 1.0f / tanf(radians * 0.5f);

        r.m[0][0] = f / aspect;
        r.m[1][1] = -f; 
        r.m[2][2] = far / (far - near);
        r.m[2][3] = 1.0f;
        r.m[3][2] = (-near * far) / (far - near);
        return r;
    }

    Matrix ortho(float left, float right, float bottom, float top, float near, float far) {
        Matrix r{};
        r.m[0][0] = 2.0f / (right - left);
        r.m[1][1] = 2.0f / (top - bottom);
        r.m[2][2] = 1.0f / (far - near);
        r.m[3][3] = 1.0f;

        r.m[3][0] = -(right + left) / (right - left);
        r.m[3][1] = -(top + bottom) / (top - bottom);
        r.m[3][2] = -near / (far - near);
        return r;
    }

   
    VkShaderModule loadShaderModule(const char* path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "Failed to open shader file: " << path << "\n";
            return VK_NULL_HANDLE;
        }
        size_t size = file.tellg();
        if (size == 0) {
            std::cerr << "Shader file empty: " << path << "\n";
            return VK_NULL_HANDLE;
        }
        // Выделить буфер в RAM и считать туда скомпилированный код .spv файлов
        std::vector<uint32_t> buffer(size / sizeof(uint32_t));
        file.seekg(0);
        // Считать в буфер содержимое шейдера
        file.read(reinterpret_cast<char*>(buffer.data()), size);
        file.close();

        VkShaderModuleCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        info.codeSize = buffer.size() * sizeof(uint32_t);
        info.pCode = buffer.data();

        VkShaderModule module = VK_NULL_HANDLE;
        if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &module) != VK_SUCCESS) {
            std::cerr << "vkCreateShaderModule failed for: " << path << "\n";
            return VK_NULL_HANDLE;
        }
        return module;
    }

    // Создание буфера и копирование данных в него
    VulkanBuffer createBuffer(size_t size, void* data, VkBufferUsageFlags usage) {
        VulkanBuffer result{};
        VkDevice& device = veekay::app.vk_device;
        VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size;
        bci.usage = usage;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bci, nullptr, &result.buffer) != VK_SUCCESS) {
            std::cerr << "Failed to create buffer\n";
            return {};
        }

        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device, result.buffer, &req);

        // Запрос у GPU информации о доступных типах памяти
        VkPhysicalDeviceMemoryProperties memProps{};
        vkGetPhysicalDeviceMemoryProperties(physical_device, &memProps);

        //Выбор подходящего типа
        const VkMemoryPropertyFlags wanted = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        uint32_t memIndex = UINT_MAX;
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if ((req.memoryTypeBits & (1u << i)) && ( (memProps.memoryTypes[i].propertyFlags & wanted) == wanted )) {
                memIndex = i;
                break;
            }
        }
        if (memIndex == UINT_MAX) {
            std::cerr << "Failed to find memory type\n";
            return {};
        }

        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = memIndex;

        // Выделяем память нужного типа и размера
        if (vkAllocateMemory(device, &mai, nullptr, &result.memory) != VK_SUCCESS) {
            std::cerr << "vkAllocateMemory failed\n";
            return {};
        }

        if (vkBindBufferMemory(device, result.buffer, result.memory, 0) != VK_SUCCESS) {
            std::cerr << "vkBindBufferMemory failed\n";
            return {};
        }

        void* dst = nullptr;
        vkMapMemory(device, result.memory, 0, req.size, 0, &dst);
        // Копируем данные из CPU в выделенный участок памяти
        memcpy(dst, data, size);
        vkUnmapMemory(device, result.memory);

        return result;
    }

    void destroyBuffer(const VulkanBuffer& b) {
        VkDevice& device = veekay::app.vk_device;
        if (b.memory) vkFreeMemory(device, b.memory, nullptr);
        if (b.buffer) vkDestroyBuffer(device, b.buffer, nullptr);
    }

    // Инициализация графического пайплайна
    void initialize() {
        VkDevice& device = veekay::app.vk_device;

        // Загружаем шейдеры (компилируются CMake'ом в spv)
        vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
        if (vertex_shader_module == VK_NULL_HANDLE) {
            veekay::app.running = false;
            return;
        }
        fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
        if (fragment_shader_module == VK_NULL_HANDLE) {
            veekay::app.running = false;
            return;
        }

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vertex_shader_module;
        stages[0].pName = "main";

        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fragment_shader_module;
        stages[1].pName = "main";
        
        VkVertexInputBindingDescription binding{};
        binding.binding = 0;
        binding.stride = sizeof(Vertex);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        // Описание атрибутов вершин
        VkVertexInputAttributeDescription attrs[2]{}; // 0 - позиция, 1 - цвет
        attrs[0].location = 0; attrs[0].binding = 0;
        attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[0].offset = offsetof(Vertex, position);

        // индекс цвета в шейдере
        attrs[1].location = 1; attrs[1].binding = 0;
        attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[1].offset = offsetof(Vertex, color);

        // Описание структуры вершин для GPU
        VkPipelineVertexInputStateCreateInfo visci{};
        visci.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        visci.vertexBindingDescriptionCount = 1;
        visci.pVertexBindingDescriptions = &binding;
        visci.vertexAttributeDescriptionCount = 2;
        visci.pVertexAttributeDescriptions = attrs;

        VkPipelineInputAssemblyStateCreateInfo iasci{};
        iasci.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        iasci.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; // вершины объединяются в треугольник
        iasci.primitiveRestartEnable = VK_FALSE;

        VkViewport viewport{};
        viewport.x = 0.0f; viewport.y = 0.0f;
        viewport.width = float(veekay::app.window_width);
        viewport.height = float(veekay::app.window_height);
        viewport.minDepth = 0.0f; viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = {veekay::app.window_width, veekay::app.window_height};

        VkPipelineViewportStateCreateInfo vp{};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1;
        vp.pViewports = &viewport;
        vp.scissorCount = 1;
        vp.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo raster{};
        raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        raster.depthClampEnable = VK_FALSE;
        raster.rasterizerDiscardEnable = VK_FALSE;
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_BACK_BIT; // отбрасываются обратные грани
        raster.frontFace = VK_FRONT_FACE_CLOCKWISE; // ориентация вершин 
        raster.depthBiasEnable = VK_FALSE;
        raster.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        ms.sampleShadingEnable = VK_FALSE;

        VkPipelineDepthStencilStateCreateInfo depth{};
        depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth.depthTestEnable = VK_TRUE;
        depth.depthWriteEnable = VK_TRUE;
        depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        depth.depthBoundsTestEnable = VK_FALSE;
        depth.stencilTestEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState att{};
        att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                             VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        att.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo blend{}; // как смешиваются цвета
        blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        blend.logicOpEnable = VK_FALSE;
        blend.attachmentCount = 1;
        blend.pAttachments = &att;

        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        push.offset = 0;
        push.size = sizeof(ShaderConstants);

        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &push;

        if (vkCreatePipelineLayout(veekay::app.vk_device, &pli, nullptr, &pipeline_layout) != VK_SUCCESS) {
            std::cerr << "Failed to create pipeline layout\n";
            veekay::app.running = false;
            return;
        }

        VkGraphicsPipelineCreateInfo gpci{};
        gpci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gpci.stageCount = 2;
        gpci.pStages = stages;
        gpci.pVertexInputState = &visci;
        gpci.pInputAssemblyState = &iasci;
        gpci.pViewportState = &vp;
        gpci.pRasterizationState = &raster;
        gpci.pMultisampleState = &ms;
        gpci.pDepthStencilState = &depth;
        gpci.pColorBlendState = &blend;
        gpci.layout = pipeline_layout;
        gpci.renderPass = veekay::app.vk_render_pass;
        gpci.subpass = 0;

        if (vkCreateGraphicsPipelines(veekay::app.vk_device, VK_NULL_HANDLE, 1, &gpci, nullptr, &pipeline) != VK_SUCCESS) {
            std::cerr << "Failed to create graphics pipeline\n";
            veekay::app.running = false;
            return;
        }

        // Генерация сферы
        const uint32_t stacks = 10; // количество горизонтальных слоёв
        const uint32_t slices = 10;
        const float radius = 1.0f;

        std::vector<Vertex> vertices;
        vertices.reserve((stacks + 1) * (slices + 1));

        for (uint32_t i = 0; i <= stacks; ++i) {
            float v = float(i) / float(stacks);
            float phi = v * float(M_PI);
            float sin_phi = sinf(phi);
            float cos_phi = cosf(phi);

            for (uint32_t j = 0; j <= slices; ++j) {
                float u = float(j) / float(slices);
                float theta = u * 2.0f * float(M_PI);
                float sin_theta = sinf(theta);
                float cos_theta = cosf(theta);

                Vector p{
                    radius * sin_phi * cos_theta,
                    radius * cos_phi,
                    radius * sin_phi * sin_theta
                };

               	Color c{
					0.5f + 0.5f * sinf(theta),        
					0.5f + 0.5f * cosf(phi),          
					0.5f + 0.5f * sinf(phi + theta)   
				};

                vertices.push_back({p, c});
            }
        }

        std::vector<uint32_t> indices;
        indices.reserve(stacks * slices * 6);
        for (uint32_t i = 0; i < stacks; ++i) {
            for (uint32_t j = 0; j < slices; ++j) {
                uint32_t i0 = i * (slices + 1) + j;
                uint32_t i1 = i0 + 1;
                uint32_t i2 = (i + 1) * (slices + 1) + j;
                uint32_t i3 = i2 + 1;

                // формируем сетку сферы
                indices.push_back(i0);
                indices.push_back(i2);
                indices.push_back(i1);

                indices.push_back(i1);
                indices.push_back(i2);
                indices.push_back(i3);
            }
        }

        g_index_count = static_cast<uint32_t>(indices.size());

        // Создаём буферы для вершин и индексов
        vertex_buffer = createBuffer(vertices.size() * sizeof(Vertex), vertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        index_buffer = createBuffer(indices.size() * sizeof(uint32_t), indices.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    }


    void shutdown() {
        VkDevice& device = veekay::app.vk_device;

        destroyBuffer(index_buffer);
        destroyBuffer(vertex_buffer);

        if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeline, nullptr);
        if (pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        if (fragment_shader_module != VK_NULL_HANDLE) vkDestroyShaderModule(device, fragment_shader_module, nullptr);
        if (vertex_shader_module != VK_NULL_HANDLE) vkDestroyShaderModule(device, vertex_shader_module, nullptr);
    }

    // обновление (GUI + анимация)

    void update(double time) {
        
        ImGui::Begin("Controls:");
        ImGui::InputFloat3("Model Translation", reinterpret_cast<float*>(&model_position));
        ImGui::SliderFloat("Model Rotation", &model_rotation, 0.0f, 2.0f * M_PI);
        ImGui::Checkbox("Spin Model", &model_spin);

        ImGui::Separator();
        ImGui::Text("Camera:");
        ImGui::SliderFloat("Yaw", &camera_yaw, -M_PI, M_PI);
        ImGui::SliderFloat("Pitch", &camera_pitch, -0.5f * M_PI, 0.5f * M_PI);
  
        ImGui::Separator();
        ImGui::Text("Animation:");
        ImGui::Checkbox("Pause", &animation_paused);
        ImGui::Checkbox("Reverse", &animation_reversed);
        ImGui::End();

        // Обновление времени анимации
        if (!animation_paused) {
            float dt = float(time) - last_update_time;
            if (dt < 0.0f) dt = 0.0f; 
            if (animation_reversed) animation_time -= dt * animation_speed;
            else animation_time += dt * animation_speed;
        }
        last_update_time = float(time);

        if (model_spin) model_rotation = animation_time;
        model_rotation = fmodf(model_rotation, 2.0f * M_PI);

        // Пульсация по синусоиде
        pulse_scale = 1.0f + 0.25f * sinf(animation_time);
    }


    void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
        vkResetCommandBuffer(cmd, 0);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        VkClearValue clear_color{};
        clear_color.color = {{0.1f, 0.1f, 0.1f, 1.0f}};
        VkClearValue clear_depth{};
        clear_depth.depthStencil = {1.0f, 0};

        VkClearValue clearValues[2] = {clear_color, clear_depth};

        VkRenderPassBeginInfo rpbi{};
        rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpbi.renderPass = veekay::app.vk_render_pass;
        rpbi.framebuffer = framebuffer;
        rpbi.renderArea.offset = {0, 0};
        rpbi.renderArea.extent = {veekay::app.window_width, veekay::app.window_height};
        rpbi.clearValueCount = 2;
        rpbi.pClearValues = clearValues;

        vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.buffer, &offset);
        vkCmdBindIndexBuffer(cmd, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

        float aspect = float(veekay::app.window_width) / float(veekay::app.window_height);
        Matrix proj = use_perspective
            ? projection(camera_fov, aspect, camera_near_plane, camera_far_plane)
            : ortho(-aspect*5.0f, aspect*5.0f, -5.0f, 5.0f, camera_near_plane, camera_far_plane);

        Matrix view = multiply(rotation_x(-camera_pitch), rotation_y(-camera_yaw));


        Matrix model = multiply(
            multiply(rotation_y(model_rotation), scale({pulse_scale, pulse_scale, pulse_scale})),
            translation(model_position)
        );

        Matrix transform = multiply(view, model);

        ShaderConstants constants{};
        constants.projection = proj;
        constants.transform = transform;

        vkCmdPushConstants(cmd, pipeline_layout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(ShaderConstants), &constants);

        if (g_index_count > 0) {
            vkCmdDrawIndexed(cmd, g_index_count, 1, 0, 0, 0);
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
