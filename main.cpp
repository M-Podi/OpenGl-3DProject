// main.cpp

#include <windows.h>
#include <stdlib.h>
#include <stdio.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "loadShaders.h" // Ensure this header declares LoadShaders

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
// NEW: Include GLM quaternion extensions
#include <glm/gtx/quaternion.hpp> 

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <map>
#include "Camera.h" // Include the updated Camera class
#include <ctime>    // Add at the top with other includes
#include <cstdlib>  // For rand() and srand()

// NEW: Include miniaudio
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

// --------------------------
// Structures to hold model data
// --------------------------
struct Mesh {
    GLuint vao;
    GLuint vbo;
    GLuint ebo;
    size_t indexCount;
    GLenum indexType; // To store the index type
    GLuint diffuse;   // Diffuse texture
    GLuint specular;  // Specular texture

    Mesh()
        : vao(0),
        vbo(0),
        ebo(0),
        indexCount(0),
        indexType(GL_UNSIGNED_SHORT),
        diffuse(0),
        specular(0) {
    }
};

// -------------------------------------------------------------------------
// UPDATED MeshInstance to store position, rotation (quat), scale, modelMatrix
// -------------------------------------------------------------------------
struct MeshInstance {
    Mesh mesh;

    // Existing transform components
    glm::vec3 position;
    glm::quat rotation;  // Quaternion for rotation
    glm::vec3 scale;

    // NEW: Add a color attribute
    glm::vec3 color;

    // Model matrix
    glm::mat4 modelMatrix;

    MeshInstance(const Mesh& m,
        const glm::vec3& pos = glm::vec3(0.0f),
        const glm::quat& rot = glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
        const glm::vec3& scl = glm::vec3(1.0f),
        const glm::vec3& col = glm::vec3(1.0f)) // Default white color
        : mesh(m)
        , position(pos)
        , rotation(rot)
        , scale(scl)
        , color(col)
        , modelMatrix(1.0f)
    {
    }
};

// --------------------------
// Global Variables
// --------------------------
std::vector<MeshInstance> modelMeshInstances; // STILL used for your original single GLB model
Mesh platformMesh;
GLuint shaderProgram;
GLuint shadowShaderProgram;

// Shadow mapping variables
const GLuint SHADOW_WIDTH = 1024, SHADOW_HEIGHT = 1024;
GLuint depthMapFBO, depthMap;

// Uniform locations
GLuint modelLoc, viewLoc, projectionLoc, lightSpaceMatrixLoc;

// Transformation matrices
glm::mat4 projectionMatrix;
glm::mat4 lightSpaceMatrix;

// Light struct
struct PointLight {
    glm::vec3 position;
    glm::vec3 ambient;
    glm::vec3 diffuse;
    glm::vec3 specular;
};

PointLight pointLight;


// Material properties for platform
struct PlatformMaterial {
    GLuint diffuse;
    GLuint specular;
    float shininess;
} platformMaterial;

// Camera
Camera camera(glm::vec3(0.0f, 5.0f, 20.0f)); // Positioned further back for visibility

// Timing
float deltaTime = 0.0f; // Time between current frame and last frame
float lastFrame = 0.0f;

// Mouse tracking
bool firstMouse = true;
float lastX = 400.0f; // Assuming window width is 800
float lastY = 300.0f; // Assuming window height is 600

// Key states
std::map<unsigned char, bool> keyStates;

// Skybox
GLuint skyboxVAO = 0;
GLuint skyboxVBO = 0;
GLuint skyboxTexture = 0;
GLuint skyboxShaderProgram = 0;

// NEW: Audio Variables
// Audio engine
ma_engine engine;

// Sound identifiers
ma_sound sound1;

// --------------------------
// Forward Declarations
// --------------------------
std::vector<MeshInstance> LoadGLBModelWithTransforms(const std::string& filename);
std::vector<MeshInstance> LoadGLBModel(const std::string& filename);
void TraverseNodes(const tinygltf::Model& model, int nodeIndex, const glm::mat4& parentTransform, std::vector<MeshInstance>& outMeshInstances);

void CreateShaders(void);
void Initialize(void);
void InitializeSkybox(void);
GLuint LoadCubemap(const std::vector<std::string>& faces);

void RenderFunction(void);
void Cleanup(void);
void keyDown(unsigned char key, int x, int y);
void keyUp(unsigned char key, int x, int y);
void mouseMove(int xpos, int ypos);
void idle();
void processInput();
void reshape(int width, int height);
void mouseClick(int button, int state, int x, int y);

// --------------------------------------------------------------------------
// NEW: Additional data structures for 3-model loading & circle arrangement
// --------------------------------------------------------------------------
std::vector<MeshInstance> modelA_Meshes;
std::vector<MeshInstance> modelB_Meshes;
std::vector<MeshInstance> modelC_Meshes;

// Final container for circle objects
std::vector<MeshInstance> sceneInstances;

// Per-instance local rotation speeds
std::vector<float> localRotationSpeeds;

// Circle rotation
float circleAngle = 0.0f;
float circleRotationSpeed = 0.5f; // Radians per second

// Forward declare our new initialization functions
void LoadAllModels();
void InitCircleObjects();

// --------------------------
// Main
// --------------------------
int main(int argc, char* argv[])
{
    srand(static_cast<unsigned int>(time(0)));

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(800, 600);
    glutCreateWindow("Project2");

    GLenum err = glewInit();
    if (GLEW_OK != err) {
        std::cerr << "Error initializing GLEW: " << glewGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    Initialize();

    // Register input callbacks
    glutKeyboardFunc(keyDown);
    glutKeyboardUpFunc(keyUp);
    glutPassiveMotionFunc(mouseMove);
    glutMouseFunc(mouseClick);

    // Register reshape callback
    glutReshapeFunc(reshape);

    // Register idle function
    glutIdleFunc(idle);

    // Hide the cursor and center it
    glutSetCursor(GLUT_CURSOR_NONE);
    glutWarpPointer(400, 300);

    glutDisplayFunc(RenderFunction);
    glutCloseFunc(Cleanup);
    glutMainLoop();

    return 0;
}

// -----------------------------------------------------
// Mouse click callback
// -----------------------------------------------------
void mouseClick(int button, int state, int x, int y)
{
    // Check if the left mouse button was pressed
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
    {
        // Print the camera's current position to the terminal
        std::cout << "Camera Position: ("
            << camera.Position.x << ", "
            << camera.Position.y << ", "
            << camera.Position.z << ")" << std::endl;
    }
}

// -----------------------------------------------------
// LoadGLBModelWithTransforms
// -----------------------------------------------------
std::vector<MeshInstance> LoadGLBModelWithTransforms(const std::string& filename)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;
    bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename);

    if (!warn.empty()) {
        std::cout << "WARNING: " << warn << std::endl;
    }
    if (!err.empty()) {
        std::cerr << "ERROR: " << err << std::endl;
    }
    if (!ret) {
        throw std::runtime_error("Failed to load glTF file: " + filename);
    }

    std::vector<MeshInstance> meshInstances;

    // Determine which scene to load
    int sceneIndex = model.defaultScene > -1 ? model.defaultScene : 0;
    if (model.scenes.empty()) {
        throw std::runtime_error("No scenes found in glTF file: " + filename);
    }

    const tinygltf::Scene& gltfScene = model.scenes[sceneIndex];

    // Traverse each node in the scene
    for (size_t i = 0; i < gltfScene.nodes.size(); ++i) {
        TraverseNodes(model, gltfScene.nodes[i], glm::mat4(1.0f), meshInstances);
    }

    return meshInstances;
}

std::vector<MeshInstance> LoadGLBModel(const std::string& filename)
{
    return LoadGLBModelWithTransforms(filename);
}

// -----------------------------------------------------
// Recursively traverse all nodes, building MeshInstance objects
// -----------------------------------------------------
void TraverseNodes(const tinygltf::Model& model, int nodeIndex,
    const glm::mat4& parentTransform,
    std::vector<MeshInstance>& outMeshInstances)
{
    const tinygltf::Node& node = model.nodes[nodeIndex];
    glm::mat4 nodeTransform = parentTransform;

    // If there's a full matrix
    if (node.matrix.size() == 16) {
        nodeTransform = nodeTransform * glm::mat4(glm::make_mat4(node.matrix.data()));
    }
    else {
        // Translation
        if (node.translation.size() == 3) {
            nodeTransform = nodeTransform *
                glm::translate(glm::mat4(1.0f),
                    glm::vec3(node.translation[0],
                        node.translation[1],
                        node.translation[2]));
        }
        // Rotation
        if (node.rotation.size() == 4) {
            glm::quat q = glm::quat(
                node.rotation[3],
                node.rotation[0],
                node.rotation[1],
                node.rotation[2]
            );
            nodeTransform = nodeTransform * glm::mat4_cast(q);
        }
        // Scale
        if (node.scale.size() == 3) {
            nodeTransform = nodeTransform *
                glm::scale(glm::mat4(1.0f),
                    glm::vec3(node.scale[0],
                        node.scale[1],
                        node.scale[2]));
        }
    }

    // If the node has a mesh, load it
    if (node.mesh >= 0) {
        const tinygltf::Mesh& gltfMesh = model.meshes[node.mesh];
        for (const auto& primitive : gltfMesh.primitives) {
            if (primitive.attributes.find("POSITION") == primitive.attributes.end()) {
                std::cerr << "Mesh primitive missing POSITION attribute. Skipping." << std::endl;
                continue;
            }
            // Positions
            const tinygltf::Accessor& posAccessor =
                model.accessors[primitive.attributes.find("POSITION")->second];
            const tinygltf::BufferView& posBufferView = model.bufferViews[posAccessor.bufferView];
            const tinygltf::Buffer& posBuffer = model.buffers[posBufferView.buffer];
            const float* positions = reinterpret_cast<const float*>(
                &posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset]);

            // Normals
            const float* normals = nullptr;
            if (primitive.attributes.find("NORMAL") != primitive.attributes.end()) {
                const tinygltf::Accessor& normAccessor =
                    model.accessors[primitive.attributes.find("NORMAL")->second];
                const tinygltf::BufferView& normBufferView = model.bufferViews[normAccessor.bufferView];
                const tinygltf::Buffer& normBuffer = model.buffers[normBufferView.buffer];
                normals = reinterpret_cast<const float*>(
                    &normBuffer.data[normBufferView.byteOffset + normAccessor.byteOffset]);
            }

            // Texture Coordinates
            const float* texCoords = nullptr;
            if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end()) {
                const tinygltf::Accessor& texAccessor =
                    model.accessors[primitive.attributes.find("TEXCOORD_0")->second];
                const tinygltf::BufferView& texBufferView = model.bufferViews[texAccessor.bufferView];
                const tinygltf::Buffer& texBuffer = model.buffers[texBufferView.buffer];
                texCoords = reinterpret_cast<const float*>(
                    &texBuffer.data[texBufferView.byteOffset + texAccessor.byteOffset]);
            }

            // Indices
            if (primitive.indices < 0) {
                std::cerr << "Primitive missing indices. Skipping." << std::endl;
                continue;
            }
            const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
            const tinygltf::BufferView& indexBufferView = model.bufferViews[indexAccessor.bufferView];
            const tinygltf::Buffer& indexBuffer = model.buffers[indexBufferView.buffer];

            GLenum indexType;
            size_t indexSize;
            if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                indexType = GL_UNSIGNED_SHORT;
                indexSize = sizeof(unsigned short);
            }
            else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                indexType = GL_UNSIGNED_INT;
                indexSize = sizeof(unsigned int);
            }
            else {
                std::cerr << "Unsupported index component type. Skipping." << std::endl;
                continue;
            }

            // Load textures
            GLuint diffuseID = 0;
            GLuint specularID = 0;
            if (primitive.material >= 0) {
                const tinygltf::Material& mat = model.materials[primitive.material];
                // baseColorTexture as diffuse
                if (mat.pbrMetallicRoughness.baseColorTexture.index >= 0) {
                    int texIndex = mat.pbrMetallicRoughness.baseColorTexture.index;
                    const tinygltf::Texture& tex = model.textures[texIndex];
                    if (tex.source < model.images.size()) {
                        const tinygltf::Image& image = model.images[tex.source];
                        glGenTextures(1, &diffuseID);
                        glBindTexture(GL_TEXTURE_2D, diffuseID);

                        GLenum format = (image.component == 3) ? GL_RGB : GL_RGBA;
                        glTexImage2D(GL_TEXTURE_2D, 0, format,
                            image.width, image.height, 0,
                            format, GL_UNSIGNED_BYTE, image.image.data());
                        glGenerateMipmap(GL_TEXTURE_2D);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
                    }
                }
                // Use normalTexture as specular or fallback to diffuse
                if (mat.normalTexture.index >= 0) {
                    int texIndex = mat.normalTexture.index;
                    const tinygltf::Texture& tex = model.textures[texIndex];
                    if (tex.source < model.images.size()) {
                        const tinygltf::Image& image = model.images[tex.source];
                        glGenTextures(1, &specularID);
                        glBindTexture(GL_TEXTURE_2D, specularID);

                        GLenum format = (image.component == 3) ? GL_RGB : GL_RGBA;
                        glTexImage2D(GL_TEXTURE_2D, 0, format,
                            image.width, image.height, 0,
                            format, GL_UNSIGNED_BYTE, image.image.data());
                        glGenerateMipmap(GL_TEXTURE_2D);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
                    }
                }
                else {
                    specularID = diffuseID;
                }
            }

            // Create interleaved vertex data
            std::vector<float> vertexData;
            vertexData.reserve(posAccessor.count * 8);
            for (size_t i = 0; i < posAccessor.count; ++i) {
                // Position
                vertexData.push_back(positions[i * 3 + 0]);
                vertexData.push_back(positions[i * 3 + 1]);
                vertexData.push_back(positions[i * 3 + 2]);

                // Normal
                if (normals) {
                    vertexData.push_back(normals[i * 3 + 0]);
                    vertexData.push_back(normals[i * 3 + 1]);
                    vertexData.push_back(normals[i * 3 + 2]);
                }
                else {
                    vertexData.push_back(0.0f);
                    vertexData.push_back(0.0f);
                    vertexData.push_back(0.0f);
                }

                // Texture Coords
                if (texCoords) {
                    vertexData.push_back(texCoords[i * 2 + 0]);
                    vertexData.push_back(texCoords[i * 2 + 1]);
                }
                else {
                    vertexData.push_back(0.0f);
                    vertexData.push_back(0.0f);
                }
            }

            // Setup OpenGL buffers
            Mesh mesh;
            mesh.indexCount = indexAccessor.count;
            mesh.indexType = indexType;

            glGenVertexArrays(1, &mesh.vao);
            glBindVertexArray(mesh.vao);

            glGenBuffers(1, &mesh.vbo);
            glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
            glBufferData(GL_ARRAY_BUFFER,
                vertexData.size() * sizeof(float),
                vertexData.data(),
                GL_STATIC_DRAW);

            glGenBuffers(1, &mesh.ebo);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ebo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                indexAccessor.count * indexSize,
                &indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset],
                GL_STATIC_DRAW);

            // Define vertex attributes
            // position attribute
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);

            // normal attribute
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));

            // texture coord attribute
            glEnableVertexAttribArray(2);
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));

            glBindVertexArray(0);

            mesh.diffuse = diffuseID;
            mesh.specular = specularID;

            // Create a MeshInstance with default transform (we'll modify later if needed)
            MeshInstance instance(mesh);
            // For now, just store nodeTransform in instance.modelMatrix (somewhat redundant):
            instance.modelMatrix = nodeTransform;

            outMeshInstances.push_back(instance);
        }
    }

    // Recursively traverse child nodes
    for (size_t i = 0; i < node.children.size(); ++i) {
        TraverseNodes(model, node.children[i], nodeTransform, outMeshInstances);
    }
}

// -----------------------------------------------------
// Shader creation
// -----------------------------------------------------
void CreateShaders(void)
{
    // Load and compile the main shader program
    shaderProgram = LoadShaders("example.vert", "example.frag");
    if (shaderProgram == 0) {
        std::cerr << "Failed to load shaders." << std::endl;
        exit(EXIT_FAILURE);
    }
    glUseProgram(shaderProgram);

    // Get uniform locations
    modelLoc = glGetUniformLocation(shaderProgram, "model");
    viewLoc = glGetUniformLocation(shaderProgram, "view");
    projectionLoc = glGetUniformLocation(shaderProgram, "projection");
    lightSpaceMatrixLoc = glGetUniformLocation(shaderProgram, "lightSpaceMatrix");
}

// -----------------------------------------------------
// LoadAllModels: NEW function to load 3 separate glb files
// -----------------------------------------------------
void LoadAllModels()
{
    try {
        // Load model A
        {
            std::string modelPathA = "normalnote.glb"; // Adjust path as needed
            std::vector<MeshInstance> temp = LoadGLBModel(modelPathA);
            modelA_Meshes.insert(modelA_Meshes.end(), temp.begin(), temp.end());
        }
        // Load model B
        {
            std::string modelPathB = "keynote.glb"; // Adjust path as needed
            std::vector<MeshInstance> temp = LoadGLBModel(modelPathB);
            modelB_Meshes.insert(modelB_Meshes.end(), temp.begin(), temp.end());
        }
        // Load model C
        {
            std::string modelPathC = "doublenote.glb"; // Adjust path as needed
            std::vector<MeshInstance> temp = LoadGLBModel(modelPathC);
            modelC_Meshes.insert(modelC_Meshes.end(), temp.begin(), temp.end());
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to load one of the models: " << e.what() << std::endl;
    }
}

// -----------------------------------------------------
// InitCircleObjects: create random circle arrangement
// -----------------------------------------------------
void InitCircleObjects()
{
    const int numObjects = 12;
    const float radius = 30.0f; // Increased radius as per previous modification

    sceneInstances.reserve(numObjects);
    localRotationSpeeds.reserve(numObjects);

    for (int i = 0; i < numObjects; i++)
    {
        // Randomly pick which model set to clone (A=0, B=1, C=2)
        int modelChoice = rand() % 3;

        // For simplicity, pick the 0th mesh from that model set
        const Mesh& chosenMesh = (modelChoice == 0)
            ? modelA_Meshes[0].mesh
            : (modelChoice == 1 ? modelB_Meshes[0].mesh : modelC_Meshes[0].mesh);

        float angle = (2.0f * 3.14159f / numObjects) * i;
        float x = radius * cos(angle);
        float z = radius * sin(angle);
        glm::vec3 pos(x, 0.0f, z);

        // Identity quaternion for start
        glm::quat rot = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);

        // Adjust the scale to make the model smaller
        float scaleFactor = 0.03f + static_cast<float>(rand()) /
            (static_cast<float>(RAND_MAX / (0.07f - 0.03f)));
        glm::vec3 scl = glm::vec3(scaleFactor);

        // Generate a random color
        glm::vec3 randomColor(
            static_cast<float>(rand()) / RAND_MAX, // Red [0.0, 1.0]
            static_cast<float>(rand()) / RAND_MAX, // Green [0.0, 1.0]
            static_cast<float>(rand()) / RAND_MAX  // Blue [0.0, 1.0]
        );

        // Construct a new instance with the random color
        MeshInstance inst(chosenMesh, pos, rot, scl, randomColor);
        sceneInstances.push_back(inst);

        // Random local spin speed: [0.5 .. 2.0]
        float spinSpeed = 0.5f + static_cast<float>(rand()) /
            (static_cast<float>(RAND_MAX / (2.0f - 0.5f)));
        localRotationSpeeds.push_back(spinSpeed);
    }
}


// -----------------------------------------------------
// Initialization
// -----------------------------------------------------
void Initialize(void)
{
    // Enable depth testing
    glEnable(GL_DEPTH_TEST);

    // Set clear color
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);

    // Compile/link shaders
    CreateShaders();

    // ---------------------
    // Load your original GLB model (if you still want it)
    // ---------------------
    std::string modelPath = "untitled.glb";
    glm::vec3 modelPosition = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 modelScale = glm::vec3(1.0f);

    try {
        std::vector<MeshInstance> loadedInstances = LoadGLBModel(modelPath);

        glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), modelScale);
        glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0f), modelPosition);
        glm::mat4 modelTransform = translationMatrix * scaleMatrix;

        // Apply the transform to each mesh instance
        for (auto& instance : loadedInstances) {
            instance.modelMatrix = modelTransform * instance.modelMatrix;
            modelMeshInstances.push_back(instance);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to load model " << modelPath << ": " << e.what() << std::endl;
    }

    // ---------------------
    // NEW: Load 3 separate models for the circle
    // ---------------------
    LoadAllModels();
    InitCircleObjects();

    // Directional light
    pointLight.position = glm::vec3(19.0f, 2.0f, 0.0f);
    pointLight.ambient = glm::vec3(0.2f, 0.2f, 0.2f);
    pointLight.diffuse = glm::vec3(0.5f, 0.5f, 0.5f);
    pointLight.specular = glm::vec3(1.0f, 1.0f, 1.0f);

    // Platform materials (if you have a platform mesh)
    platformMaterial.diffuse = platformMesh.diffuse;
    platformMaterial.specular = platformMesh.specular;
    platformMaterial.shininess = 16.0f;

    // Shadow Mapping setup
    {
        glGenFramebuffers(1, &depthMapFBO);
        glGenTextures(1, &depthMap);
        glBindTexture(GL_TEXTURE_2D, depthMap);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
            SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        float borderColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

        glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        shadowShaderProgram = LoadShaders("shadow_shader.vert", "shadow_shader.frag");
        if (shadowShaderProgram == 0) {
            std::cerr << "Failed to load shadow shaders." << std::endl;
            exit(EXIT_FAILURE);
        }

        float near_plane = 1.0f;
        float far_plane = 100.0f;  // sau cât de mare e scena ta
        glm::mat4 lightProjection = glm::perspective(glm::radians(90.0f), 1.0f, near_plane, far_plane);

        // Poziția luminii, deja setată: pointLight.position = (10,10,10), de ex.
        glm::mat4 lightView = glm::lookAt(
            pointLight.position,       // de unde privim
            glm::vec3(0.0f, 0.0f, 0.0f), // țintă (centrul scenei)
            glm::vec3(0.0f, 1.0f, 0.0f)  // up-vector
        );

        lightSpaceMatrix = lightProjection * lightView;

        // Pass to shadow shader
        glUseProgram(shadowShaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(shadowShaderProgram, "lightSpaceMatrix"),
            1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));

        // Pass to main shader
        glUseProgram(shaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "lightSpaceMatrix"),
            1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));

        // Shadow map unit = 2
        glUniform1i(glGetUniformLocation(shaderProgram, "shadowMap"), 2);
    }

    // Set up the camera projection
    glm::mat4 viewMatrix = camera.GetViewMatrix();
    projectionMatrix = glm::perspective(glm::radians(camera.Zoom),
        800.0f / 600.0f,
        0.1f, 100.0f);

    glUseProgram(shaderProgram);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"),
        1, GL_FALSE, glm::value_ptr(viewMatrix));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"),
        1, GL_FALSE, glm::value_ptr(projectionMatrix));

    // Set light properties
    glUniform3fv(glGetUniformLocation(shaderProgram, "pointLight.position"), 1, glm::value_ptr(pointLight.position));
    glUniform3fv(glGetUniformLocation(shaderProgram, "pointLight.ambient"), 1, glm::value_ptr(pointLight.ambient));
    glUniform3fv(glGetUniformLocation(shaderProgram, "pointLight.diffuse"), 1, glm::value_ptr(pointLight.diffuse));
    glUniform3fv(glGetUniformLocation(shaderProgram, "pointLight.specular"), 1, glm::value_ptr(pointLight.specular));


    // Set camera position
    GLuint viewPosLoc = glGetUniformLocation(shaderProgram, "viewPos");
    glUniform3fv(viewPosLoc, 1, glm::value_ptr(camera.Position));

    // Initialize miniaudio engine
    if (ma_engine_init(NULL, &engine) != MA_SUCCESS) {
        std::cerr << "Failed to initialize audio engine." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Load and play
    if (ma_engine_play_sound(&engine, "exile2.wav", 0) != MA_SUCCESS) {
        std::cerr << "Failed to load sound1.mp3" << std::endl;
    }
    else {
        ma_sound_set_looping(&sound1, MA_TRUE);
    }

    

    // -----------------------------
    // ------------- SKYBOX INIT -------------
    // -----------------------------
    InitializeSkybox();
}

// -----------------------------------------------------
// Initialize Skybox
// -----------------------------------------------------
void InitializeSkybox(void)
{
    // Skybox vertices (cube)
    float skyboxVertices[] = {
        // positions
        -1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

        -1.0f,  1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f,  1.0f,
         1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f
    };

    // Setup VAO/VBO
    glGenVertexArrays(1, &skyboxVAO);
    glGenBuffers(1, &skyboxVBO);
    glBindVertexArray(skyboxVAO);
    glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices),
        &skyboxVertices, GL_STATIC_DRAW);

    // Position layout
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
        3 * sizeof(float), (void*)0);

    glBindVertexArray(0);

    // Load the cubemap textures
    std::vector<std::string> faces
    {
        "skybox2/px.png",
        "skybox2/nx.png",
        "skybox2/py.png",
        "skybox2/ny.png",
        "skybox2/pz.png",
        "skybox2/nz.png"
    };
    skyboxTexture = LoadCubemap(faces);

    // Create skybox shader
    skyboxShaderProgram = LoadShaders("skybox.vert", "skybox.frag");
    if (skyboxShaderProgram == 0) {
        std::cerr << "Failed to load skybox shaders." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Use the skybox shader and set uniform
    glUseProgram(skyboxShaderProgram);
    glUniform1i(glGetUniformLocation(skyboxShaderProgram, "skybox"), 0);
}

// -----------------------------------------------------
// LoadCubemap
// -----------------------------------------------------
GLuint LoadCubemap(const std::vector<std::string>& faces)
{
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

    for (GLuint i = 0; i < faces.size(); i++)
    {
        int width, height, nrChannels;
        unsigned char* data = stbi_load(faces[i].c_str(), &width, &height, &nrChannels, 0);
        if (data)
        {
            GLenum format = (nrChannels == 4) ? GL_RGBA : GL_RGB;
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                0, format, width, height,
                0, format, GL_UNSIGNED_BYTE, data);
            stbi_image_free(data);
        }
        else
        {
            std::cerr << "Cubemap texture failed to load at path: "
                << faces[i] << std::endl;
            stbi_image_free(data);
        }
    }

    // Set texture parameters
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // Prevent seams
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    return textureID;
}

// -----------------------------------------------------
// Render Function
// -----------------------------------------------------
void RenderFunction(void)
{
    // Calculate delta time
    float currentFrame = (float)glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    // Process input
    processInput();

    // --------------------------------------------
    // NEW: Update circle angle & local rotations
    // --------------------------------------------
    circleAngle += circleRotationSpeed * deltaTime;
    glm::quat circleRotation = glm::angleAxis(circleAngle, glm::vec3(0.0f, 1.0f, 0.0f));

    for (size_t i = 0; i < sceneInstances.size(); i++)
    {
        // Spin each object around its own Y axis
        float localAngle = localRotationSpeeds[i] * deltaTime;
        glm::quat localSpin = glm::angleAxis(localAngle, glm::vec3(0.0f, 1.0f, 0.0f));
        sceneInstances[i].rotation = localSpin * sceneInstances[i].rotation;

        // Build final modelMatrix:
        glm::mat4 groupRotMatrix = glm::toMat4(circleRotation);
        glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0f), sceneInstances[i].position);
        glm::mat4 localRotMatrix = glm::toMat4(sceneInstances[i].rotation);
        glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), sceneInstances[i].scale);

        // Combine: group rotation -> translation -> local rotation -> scale
        sceneInstances[i].modelMatrix = groupRotMatrix * translationMatrix * localRotMatrix * scaleMatrix;
    }

    // Update camera's view matrix
    glm::mat4 view = camera.GetViewMatrix();
    glUseProgram(shaderProgram);
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));

    // 1. Shadow Pass
    glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
    glClear(GL_DEPTH_BUFFER_BIT);
    glUseProgram(shadowShaderProgram);
    glUniformMatrix4fv(glGetUniformLocation(shadowShaderProgram, "lightSpaceMatrix"),
        1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));

    // Render original model meshInstances to shadow map
    for (const auto& instance : modelMeshInstances) {
        glUniformMatrix4fv(glGetUniformLocation(shadowShaderProgram, "model"),
            1, GL_FALSE, glm::value_ptr(instance.modelMatrix));
        glBindVertexArray(instance.mesh.vao);
        glDrawElements(GL_TRIANGLES,
            (GLsizei)instance.mesh.indexCount,
            instance.mesh.indexType,
            0);
        glBindVertexArray(0);
    }

    // Render circle model objects to shadow map
    for (const auto& instance : sceneInstances) {
        glUniformMatrix4fv(glGetUniformLocation(shadowShaderProgram, "model"),
            1, GL_FALSE, glm::value_ptr(instance.modelMatrix));
        glBindVertexArray(instance.mesh.vao);
        glDrawElements(GL_TRIANGLES,
            (GLsizei)instance.mesh.indexCount,
            instance.mesh.indexType,
            0);
        glBindVertexArray(0);
    }

    // If you have a platform, render that to shadow map as well...
    // (omitted for brevity)

    // Render ground plane to shadow map
    glm::mat4 groundModel = glm::mat4(1.0f); // Identity matrix for ground at origin
    glUniformMatrix4fv(glGetUniformLocation(shadowShaderProgram, "model"),
        1, GL_FALSE, glm::value_ptr(groundModel));
    glBindVertexArray(platformMesh.vao); // Assuming platformMesh is your ground plane
    glDrawElements(GL_TRIANGLES, (GLsizei)platformMesh.indexCount, platformMesh.indexType, 0);
    glBindVertexArray(0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // 2. Render Pass
    int windowWidth = 800;
    int windowHeight = 600;
    glViewport(0, 0, windowWidth, windowHeight);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(shaderProgram);

    // Bind shadow map (texture unit 2)
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, depthMap);

    // Update camera pos
    GLuint viewPosLoc = glGetUniformLocation(shaderProgram, "viewPos");
    glUniform3fv(viewPosLoc, 1, glm::value_ptr(camera.Position));

    // Projection matrix is global, but let's ensure uniform is updated
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projectionMatrix));

    // Render platform (ground plane)
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(groundModel));
    glUniform1i(glGetUniformLocation(shaderProgram, "material.diffuse"), 0);
    glUniform1i(glGetUniformLocation(shaderProgram, "material.specular"), 1);
    glUniform1f(glGetUniformLocation(shaderProgram, "material.shininess"), 32.0f);

    if (platformMesh.diffuse != 0) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, platformMesh.diffuse);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D,
            platformMesh.specular != 0 ? platformMesh.specular : platformMesh.diffuse);
    }

    glBindVertexArray(platformMesh.vao);
    glDrawElements(GL_TRIANGLES,
        (GLsizei)platformMesh.indexCount,
        platformMesh.indexType,
        0);
    glBindVertexArray(0);

    // -----------------------------------------
    // Render original loaded GLB model
    // -----------------------------------------
    for (const auto& instance : modelMeshInstances) {
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(instance.modelMatrix));
        glUniform1i(glGetUniformLocation(shaderProgram, "material.diffuse"), 0);
        glUniform1i(glGetUniformLocation(shaderProgram, "material.specular"), 1);
        glUniform1f(glGetUniformLocation(shaderProgram, "material.shininess"), 32.0f);

        if (instance.mesh.diffuse != 0) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, instance.mesh.diffuse);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D,
                instance.mesh.specular != 0 ? instance.mesh.specular : instance.mesh.diffuse);
        }

        glBindVertexArray(instance.mesh.vao);
        glDrawElements(GL_TRIANGLES,
            (GLsizei)instance.mesh.indexCount,
            instance.mesh.indexType,
            0);
        glBindVertexArray(0);
    }

    // -----------------------------------------
    // Render our circle objects
    // -----------------------------------------
    for (const auto& instance : sceneInstances) {
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(instance.modelMatrix));
        glUniform1i(glGetUniformLocation(shaderProgram, "material.diffuse"), 0);
        glUniform1i(glGetUniformLocation(shaderProgram, "material.specular"), 1);
        glUniform1f(glGetUniformLocation(shaderProgram, "material.shininess"), 32.0f);

        if (instance.mesh.diffuse != 0) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, instance.mesh.diffuse);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D,
                instance.mesh.specular != 0 ? instance.mesh.specular : instance.mesh.diffuse);
        }

        glBindVertexArray(instance.mesh.vao);
        glDrawElements(GL_TRIANGLES,
            (GLsizei)instance.mesh.indexCount,
            instance.mesh.indexType,
            0);
        glBindVertexArray(0);
    }

    // -----------------------------------------
    // Render Skybox last
    // -----------------------------------------
    glDepthFunc(GL_LEQUAL);
    glUseProgram(skyboxShaderProgram);

    glm::mat4 skyboxView = glm::mat4(glm::mat3(camera.GetViewMatrix()));
    GLuint skyboxViewLoc = glGetUniformLocation(skyboxShaderProgram, "view");
    GLuint skyboxProjLoc = glGetUniformLocation(skyboxShaderProgram, "projection");
    glUniformMatrix4fv(skyboxViewLoc, 1, GL_FALSE, glm::value_ptr(skyboxView));
    glUniformMatrix4fv(skyboxProjLoc, 1, GL_FALSE, glm::value_ptr(projectionMatrix));

    glBindVertexArray(skyboxVAO);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, skyboxTexture);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);

    glDepthFunc(GL_LESS);

    glutSwapBuffers();
}

// -----------------------------------------------------
// Cleanup
// -----------------------------------------------------
void Cleanup(void)
{
    // Stop and uninitialize sounds
    ma_sound_uninit(&sound1);

    // Uninitialize audio engine
    ma_engine_uninit(&engine);

    glDeleteProgram(shaderProgram);
    glDeleteProgram(shadowShaderProgram);

    // Cleanup original model mesh instances
    for (auto& instance : modelMeshInstances) {
        glDeleteVertexArrays(1, &instance.mesh.vao);
        glDeleteBuffers(1, &instance.mesh.vbo);
        glDeleteBuffers(1, &instance.mesh.ebo);
        if (instance.mesh.diffuse) {
            glDeleteTextures(1, &instance.mesh.diffuse);
        }
        if (instance.mesh.specular) {
            glDeleteTextures(1, &instance.mesh.specular);
        }
    }

    // Cleanup circle objects
    for (auto& instance : sceneInstances) {
        glDeleteVertexArrays(1, &instance.mesh.vao);
        glDeleteBuffers(1, &instance.mesh.vbo);
        glDeleteBuffers(1, &instance.mesh.ebo);
        if (instance.mesh.diffuse) {
            glDeleteTextures(1, &instance.mesh.diffuse);
        }
        if (instance.mesh.specular) {
            glDeleteTextures(1, &instance.mesh.specular);
        }
    }

    // Cleanup the triple models if they differ from sceneInstances 
    // (Here we assume each sub-vector is not re-used in sceneInstances 
    //  but if they share the same VAO, be sure not to double-free).
    // For safety, do similar cleanup for modelA_Meshes, etc. if needed.

    // Cleanup platform
    glDeleteVertexArrays(1, &platformMesh.vao);
    glDeleteBuffers(1, &platformMesh.vbo);
    glDeleteBuffers(1, &platformMesh.ebo);
    if (platformMesh.diffuse) {
        glDeleteTextures(1, &platformMesh.diffuse);
    }
    if (platformMesh.specular) {
        glDeleteTextures(1, &platformMesh.specular);
    }

    // Cleanup shadow resources
    glDeleteFramebuffers(1, &depthMapFBO);
    glDeleteTextures(1, &depthMap);

    // Cleanup skybox
    glDeleteVertexArrays(1, &skyboxVAO);
    glDeleteBuffers(1, &skyboxVBO);
    glDeleteTextures(1, &skyboxTexture);
    glDeleteProgram(skyboxShaderProgram);
}

// -----------------------------------------------------
// Keyboard input
// -----------------------------------------------------
void keyDown(unsigned char key, int x, int y)
{
    keyStates[key] = true;
    if (key == 27) { // ESC
        Cleanup();
        exit(0);
    }
}

void keyUp(unsigned char key, int x, int y)
{
    keyStates[key] = false;
}

// -----------------------------------------------------
// Mouse movement (FPS-style)
// -----------------------------------------------------
void mouseMove(int xpos, int ypos)
{
    int windowWidth = 800;
    int windowHeight = 600;

    if (firstMouse)
    {
        lastX = (float)xpos;
        lastY = (float)ypos;
        firstMouse = false;
    }

    float xoffset = (float)(xpos - windowWidth / 2);
    float yoffset = (float)(windowHeight / 2 - ypos);

    lastX = (float)windowWidth / 2.0f;
    lastY = (float)windowHeight / 2.0f;

    camera.ProcessMouseMovement(xoffset, yoffset);

    glutWarpPointer(windowWidth / 2, windowHeight / 2);
}

// -----------------------------------------------------
// Process Input
// -----------------------------------------------------
void processInput()
{
    if (keyStates['w'] || keyStates['W'])
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (keyStates['s'] || keyStates['S'])
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (keyStates['a'] || keyStates['A'])
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (keyStates['d'] || keyStates['D'])
        camera.ProcessKeyboard(RIGHT, deltaTime);
}

// -----------------------------------------------------
// Idle
// -----------------------------------------------------
void idle()
{
    glutPostRedisplay();
}

// -----------------------------------------------------
// Reshape
// -----------------------------------------------------
void reshape(int width, int height)
{
    glViewport(0, 0, width, height);
    projectionMatrix = glm::perspective(glm::radians(camera.Zoom),
        (float)width / (float)height,
        0.1f, 100.0f);
    glUseProgram(shaderProgram);
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projectionMatrix));

    // Update light space matrix if necessary
    float near_plane = 1.0f;
    float far_plane = 100.0f;  // sau cât de mare e scena ta
    glm::mat4 lightProjection = glm::perspective(glm::radians(90.0f), 1.0f, near_plane, far_plane);

    // Poziția luminii, deja setată: pointLight.position = (10,10,10), de ex.
    glm::mat4 lightView = glm::lookAt(
        pointLight.position,       // de unde privim
        glm::vec3(0.0f, 0.0f, 0.0f), // țintă (centrul scenei)
        glm::vec3(0.0f, 1.0f, 0.0f)  // up-vector
    );

    lightSpaceMatrix = lightProjection * lightView;

    // Update shadow shader
    glUseProgram(shadowShaderProgram);
    glUniformMatrix4fv(glGetUniformLocation(shadowShaderProgram, "lightSpaceMatrix"),
        1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));

    // Update main shader
    glUseProgram(shaderProgram);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "lightSpaceMatrix"),
        1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));
}