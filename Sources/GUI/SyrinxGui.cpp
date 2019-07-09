#include "SyrinxGui.h"

namespace Syrinx {

Gui::Gui(FileManager *fileManager, ShaderManager *shaderManager, HardwareResourceManager *hardwareResourceManager)
    : mFileManager(fileManager)
    , mShaderManager(shaderManager)
    , mHardwareResourceManager(hardwareResourceManager)
{
    SYRINX_ENSURE(mFileManager);
    SYRINX_ENSURE(mShaderManager);
    SYRINX_ENSURE(mHardwareResourceManager);
    SYRINX_ENSURE(!mActiveFont);
}


Gui::~Gui()
{
    ImGui::DestroyContext();
}


void Gui::init()
{
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    mapKey();
    ImGui::GetIO().IniFilename = nullptr;
    createRenderResource();
}


void Gui::beginFrame()
{
    ImGui::NewFrame();
}


void Gui::onWindowResize(uint32_t width, uint32_t height)
{
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize.x = static_cast<float>(width);
    io.DisplaySize.y = static_cast<float>(height);
}


void Gui::onInputEvents(Input *input)
{
    if (!input) {
        return;
    }

    static bool mouseJustPressed[5] = {false, false, false, false, false};

    ImGuiIO& io = ImGui::GetIO();
    for (int i = 0; i < IM_ARRAYSIZE(io.MouseDown) && i < MouseBotton::_size_constant; ++ i) {
        io.MouseDown[i] = mouseJustPressed[i] | input->getMouseAction(MouseBotton::_from_index(i));
        mouseJustPressed[i] = false;
    }

    const ImVec2 mousePosBackup = io.MousePos;
    io.MousePos = ImVec2(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());

    const bool focused = input->isFocused();
    if (focused) {
        if (io.WantSetMousePos) {
            input->setMousePos(mousePosBackup.x, mousePosBackup.y);
        } else {
            auto cursorPos = input->getCursorPosition();
            io.MousePos = ImVec2(cursorPos.x, cursorPos.y);
        }
    }
}


void Gui::mapKey()
{
    ImGuiIO& io = ImGui::GetIO();

    io.BackendFlags |= ImGuiBackendFlags_HasMouseCursors;
    io.BackendFlags |= ImGuiBackendFlags_HasSetMousePos;

    io.KeyMap[ImGuiKey_Tab] = GLFW_KEY_TAB;
    io.KeyMap[ImGuiKey_LeftArrow] = GLFW_KEY_LEFT;
    io.KeyMap[ImGuiKey_RightArrow] = GLFW_KEY_RIGHT;
    io.KeyMap[ImGuiKey_UpArrow] = GLFW_KEY_UP;
    io.KeyMap[ImGuiKey_DownArrow] = GLFW_KEY_DOWN;
    io.KeyMap[ImGuiKey_PageUp] = GLFW_KEY_PAGE_UP;
    io.KeyMap[ImGuiKey_PageDown] = GLFW_KEY_PAGE_DOWN;
    io.KeyMap[ImGuiKey_Home] = GLFW_KEY_HOME;
    io.KeyMap[ImGuiKey_End] = GLFW_KEY_END;
    io.KeyMap[ImGuiKey_Insert] = GLFW_KEY_INSERT;
    io.KeyMap[ImGuiKey_Delete] = GLFW_KEY_DELETE;
    io.KeyMap[ImGuiKey_Backspace] = GLFW_KEY_BACKSPACE;
    io.KeyMap[ImGuiKey_Space] = GLFW_KEY_SPACE;
    io.KeyMap[ImGuiKey_Enter] = GLFW_KEY_ENTER;
    io.KeyMap[ImGuiKey_Escape] = GLFW_KEY_ESCAPE;
    io.KeyMap[ImGuiKey_A] = GLFW_KEY_A;
    io.KeyMap[ImGuiKey_C] = GLFW_KEY_C;
    io.KeyMap[ImGuiKey_V] = GLFW_KEY_V;
    io.KeyMap[ImGuiKey_X] = GLFW_KEY_X;
    io.KeyMap[ImGuiKey_Y] = GLFW_KEY_Y;
    io.KeyMap[ImGuiKey_Z] = GLFW_KEY_Z;
}


void Gui::createRenderResource()
{
    SYRINX_EXPECT(!mVertexInputState);
    mShader = mShaderManager->createOrRetrieve("syrinx-gui.shader");

    createVertexInputState();
    buildFontTexture();
    createRenderState();

    SYRINX_ENSURE(mShader);
    SYRINX_ENSURE(mVertexInputState);
}


void Gui::createRenderState()
{
    SYRINX_EXPECT(mVertexInputState);
    SYRINX_EXPECT(mShader);

    mRenderState.viewportState.enableScissor = true;
    mRenderState.depthStencilState.enableDepthTest = false;
    mRenderState.colorBlendState.setBlendEnable(0, true)
        .setColorBlendFunc(0, BlendFactor::SrcAlpha, BlendOp::Add, BlendFactor::OneMinusSrcAlpha);

    mRenderState.setVertexInputState(mVertexInputState);
    mRenderState.setProgramPipeline(mShader->getProgramPipeline());
}


void Gui::buildFontTexture()
{
    ImGuiIO& io = ImGui::GetIO();
    unsigned char *pixels = nullptr;
    int width = 0, height = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    SYRINX_ASSERT(pixels);

    if (mFontTexture) {
        mHardwareResourceManager->destroyHardwareTexture(mFontTexture->getName());
    }

    if (mSampledFontTexture) {
        auto textureView = mSampledFontTexture->getTextureView();
        auto sampler = mSampledFontTexture->getSampler();
        mHardwareResourceManager->destroyHardwareTextureView(textureView.getName());
        mHardwareResourceManager->destroyHardwareSampler(sampler.getName());
    }

    mFontTexture = mHardwareResourceManager->createTexture2D("SyrinxGUI-FontTex", PixelFormat::RGBA8, static_cast<uint32_t>(width), static_cast<uint32_t>(height), false);
    mFontTexture->write(pixels, width, height);
    io.Fonts->TexID = reinterpret_cast<ImTextureID>(mFontTexture->getHandle());

    TextureViewDesc textureViewDesc;
    textureViewDesc.type = TextureType::TEXTURE_2D;
    textureViewDesc.levelCount = mFontTexture->getMaxMipMapLevel();
    auto textureView = mHardwareResourceManager->createTextureView("SyrinxGuiFontTextureView", mFontTexture, textureViewDesc);

    SamplingSetting  samplingSetting;
    samplingSetting.setMinFilterMethod(TextureMinFilterMethod::LINEAR);
    samplingSetting.setMagFilterMethod(TextureMagFilterMethod::LINEAR);
    samplingSetting.setWrapRMethod(TextureWrapMethod::CLAMP_TO_BORDER);
    samplingSetting.setWrapSMethod(TextureWrapMethod::CLAMP_TO_BORDER);
    samplingSetting.setWrapTMethod(TextureWrapMethod::CLAMP_TO_BORDER);
    samplingSetting.setBorderColor(Color(0.0f, 0.0f, 0.0f, 1.0f));

    auto fontSampler = mHardwareResourceManager->createSampler("SyrinxGuiFontSampler", samplingSetting);
    mSampledFontTexture = std::make_unique<SampledTexture>(*textureView, *fontSampler);
}


void Gui::render(RenderContext *renderContext)
{
    SYRINX_EXPECT(renderContext);

    ImGui::Render();
    auto guiDrawData = ImGui::GetDrawData();
    SYRINX_ASSERT(guiDrawData);

    int framebufferWidth = static_cast<int>(guiDrawData->DisplaySize.x);
    int framebufferHeight = static_cast<int>(guiDrawData->DisplaySize.y);
    if (framebufferWidth == 0 || framebufferHeight == 0) {
        return;
    }

    mRenderState.viewportState.viewport.offset = Offset2D<uint32_t>(0, 0);
    mRenderState.viewportState.viewport.extent = Extent2D<uint32_t>(framebufferWidth, framebufferHeight);

    float L = guiDrawData->DisplayPos.x;
    float R = guiDrawData->DisplayPos.x + guiDrawData->DisplaySize.x;
    float T = guiDrawData->DisplayPos.y;
    float B = guiDrawData->DisplayPos.y + guiDrawData->DisplaySize.y;
    const Matrix4x4 orthoProjectionMat = glm::ortho(L, R, B, T, -1.0f, 1.0f);

    auto vertexModule = mShader->getShaderModule(ProgramStageType::VertexStage);
    auto fragmentModule = mShader->getShaderModule(ProgramStageType::FragmentStage);

    auto& vertexProgVars = *(vertexModule->getProgramVars());
    auto& fragmentProgVars = *(fragmentModule->getProgramVars());

    vertexProgVars["SyrinxGuiMatrixBuffer"]["SYRINX_MATRIX_PROJ"] = orthoProjectionMat;
    vertexModule->updateProgramVars(vertexProgVars);
    vertexModule->uploadParametersToGpu();
    vertexModule->bindResources();

    fragmentProgVars.setTexture("inTex", mSampledFontTexture.get());
    fragmentModule->updateProgramVars(fragmentProgVars);
    fragmentModule->uploadParametersToGpu();
    fragmentModule->bindResources();

    renderContext->pushRenderState();
    renderContext->setRenderState(&mRenderState);
    for (int i = 0; i < guiDrawData->CmdListsCount; ++i) {
        const ImDrawList *cmdList = guiDrawData->CmdLists[i];
        mVertexBuffer = createVertexBuffer(cmdList->VtxBuffer);
        mIndexBuffer = createIndexBuffer(cmdList->IdxBuffer);
        mVertexInputState->setVertexBuffer(0, mVertexBuffer);
        mVertexInputState->setIndexBuffer(mIndexBuffer);
        mVertexInputState->setup();

        renderContext->prepareDraw();
        for (int cmdIndex = 0; cmdIndex < cmdList->CmdBuffer.Size; ++ cmdIndex) {
            const ImDrawCmd *drawCmd = &(cmdList->CmdBuffer[cmdIndex]);
            renderContext->drawIndexed(drawCmd->ElemCount, drawCmd->IdxOffset * sizeof(ImDrawIdx));
        }
    }
    renderContext->popRenderState();
}


void Gui::addFont(const std::string& name, const std::string& fileName)
{
    if (mFontMap.find(name) != std::end(mFontMap)) {
        return;
    }

    auto [fileExist, fullPath] = mFileManager->findFile(fileName);
    if (!fileExist) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
            "fail to add font [name={}, file name={}] into GUI because the file name can not be found", name, fileName);
    }
    constexpr float fontSize = 14.0f;
    ImFont *font = ImGui::GetIO().Fonts->AddFontFromFileTTF(fullPath.c_str(), fontSize);
    SYRINX_ASSERT(font);
    mFontMap[name] = font;
    buildFontTexture();
    SYRINX_ENSURE(mFontMap.find(name) != std::end(mFontMap));
}


void Gui::setActiveFont(const std::string& fontName)
{
    auto iter = mFontMap.find(fontName);
    if (iter == std::end(mFontMap)) {
        mActiveFont = nullptr;
    } else {
        mActiveFont = iter->second;
    }
}


ImFont* Gui::getActiveFont() const
{
    return mActiveFont;
}


void Gui::createVertexInputState()
{
    SYRINX_EXPECT(!mVertexInputState);
    mVertexInputState = mHardwareResourceManager->createVertexInputState("SyrinxGuiVertexInpuState");

    VertexAttributeLayoutDesc vertexAttributeLayoutDesc;

    VertexAttributeDescription positionAttributeDesc;
    positionAttributeDesc.setSemantic(VertexAttributeSemantic::Position);
    positionAttributeDesc.setLocation(0);
    positionAttributeDesc.setDataType(VertexAttributeDataType::FLOAT2);
    positionAttributeDesc.setDataOffset(IM_OFFSETOF(ImDrawVert, pos));
    positionAttributeDesc.setBindingPoint(0);

    VertexAttributeDescription uvAttributeDesc;
    uvAttributeDesc.setSemantic(VertexAttributeSemantic::TexCoord);
    uvAttributeDesc.setLocation(1);
    uvAttributeDesc.setDataType(VertexAttributeDataType::FLOAT2);
    uvAttributeDesc.setDataOffset(IM_OFFSETOF(ImDrawVert, uv));
    uvAttributeDesc.setBindingPoint(0);

    VertexAttributeDescription colorAttributeDesc;
    colorAttributeDesc.setSemantic(VertexAttributeSemantic::Color);
    colorAttributeDesc.setLocation(2);
    colorAttributeDesc.setDataType(VertexAttributeDataType::UBYTE4);
    colorAttributeDesc.setDataOffset(IM_OFFSETOF(ImDrawVert, col));
    colorAttributeDesc.setBindingPoint(0);

    vertexAttributeLayoutDesc.addVertexAttributeDesc(positionAttributeDesc);
    vertexAttributeLayoutDesc.addVertexAttributeDesc(uvAttributeDesc);
    vertexAttributeLayoutDesc.addVertexAttributeDesc(colorAttributeDesc);

    mVertexInputState->setVertexAttributeLayoutDesc(std::move(vertexAttributeLayoutDesc));
    SYRINX_ENSURE(mVertexInputState);
}


HardwareVertexBuffer* Gui::createVertexBuffer(const ImVector<ImDrawVert>& vertexData) const
{
    constexpr char vertexBufferName[] = "SyrinxGuiVertexBuffer";

    if (!mVertexBuffer) {
        return mHardwareResourceManager->createVertexBuffer(vertexBufferName, vertexData.Size, sizeof(ImDrawVert), vertexData.Data);
    }

    size_t requiredSize = vertexData.Size * sizeof(ImDrawVert);
    auto vertexBuffer = mVertexBuffer;
    if (mVertexBuffer->getSize() < requiredSize) {
        mHardwareResourceManager->destroyHardwareVertexBuffer(vertexBufferName);
        vertexBuffer = mHardwareResourceManager->createVertexBuffer(vertexBufferName, 2 * vertexData.Size, sizeof(ImDrawVert));
    }
    vertexBuffer->setData(vertexData.Data, vertexData.Size);
    vertexBuffer->uploadToGpu(0, requiredSize);
    return vertexBuffer;
}


HardwareIndexBuffer* Gui::createIndexBuffer(const ImVector<ImDrawIdx>& indexData) const
{
    constexpr char indexBufferName[] = "SyrinxGuiIndexBuffer";
    const IndexType indexType = (sizeof(ImDrawIdx) == 2) ? IndexType::UINT16 : IndexType::UINT32;

    if (!mIndexBuffer) {
        return mHardwareResourceManager->createIndexBuffer(indexBufferName, indexData.Size, indexType, indexData.Data);
    }

    size_t requiredSize = indexData.Size * sizeof(ImDrawIdx);
    auto indexBuffer = mIndexBuffer;
    if (requiredSize > mIndexBuffer->getSize()) {
        mHardwareResourceManager->destroyHardwareIndexBuffer(indexBufferName);
        indexBuffer = mHardwareResourceManager->createIndexBuffer(indexBufferName, 2 * indexData.Size, indexType);
    }
    indexBuffer->setData(indexData.Data, indexData.Size);
    indexBuffer->uploadToGpu(0, requiredSize);
    return indexBuffer;
}

} // namespace Syrinx
