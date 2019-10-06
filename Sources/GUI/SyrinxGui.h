#pragma once
#include <memory>
#include <imgui/imgui.h>
#include <Input/SyrinxInput.h>
#include <Graphics/SyrinxRenderState.h>
#include <Graphics/SyrinxRenderContext.h>
#include <Pipeline/SyrinxRenderWindow.h>
#include <ResourceManager/SyrinxShaderManager.h>
#include <Manager/SyrinxHardwareResourceManager.h>

namespace Syrinx {

class Gui {
public:
    Gui(FileManager *fileManager, ShaderManager *shaderManager, HardwareResourceManager *hardwareResourceManager);
    ~Gui();
    void init();
    void beginFrame();
    void onInputEvents(Input *input);
    void onWindowResize(uint32_t width, uint32_t height);
    void render(RenderContext *renderContext);
    void addFont(const std::string& name, const std::string& fileName);
    void setActiveFont(const std::string& fontName);
    ImFont* getActiveFont() const;

private:
    void mapKey();
    void createRenderResource();
    void createRenderState();
    void buildFontTexture();
    void createVertexInputState();
    HardwareVertexBuffer* createVertexBuffer(const ImVector<ImDrawVert>& vertexData) const;
    HardwareIndexBuffer* createIndexBuffer(const ImVector<ImDrawIdx>& indexData) const;

private:
    FileManager *mFileManager = nullptr;
    ShaderManager *mShaderManager = nullptr;
    HardwareResourceManager *mHardwareResourceManager = nullptr;
    Shader *mShader = nullptr;
    HardwareTexture *mFontTexture = nullptr;
    SampledTexture mSampledFontTexture;
    VertexInputState *mVertexInputState = nullptr;
    HardwareVertexBuffer *mVertexBuffer = nullptr;
    HardwareIndexBuffer *mIndexBuffer = nullptr;
    RenderState mRenderState;
    std::map<std::string, ImFont*> mFontMap;
    ImFont *mActiveFont = nullptr;
};

} // namespace Syrinx