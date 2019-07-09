#include "SyrinxRenderTarget.h"
#include <Logging/SyrinxLogManager.h>
#include <Exception/SyrinxException.h>

namespace Syrinx {

const Color RenderTarget::DEFAULT_CLEAR_COLOR_VALUE = {0.5, 0.0, 0.5, 1.0};

int RenderTarget::getMaxColorAttachmentCount()
{
    GLint maxAttachmentCount = 0;
    glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &maxAttachmentCount);
    SYRINX_ASSERT(maxAttachmentCount > 0);
    return maxAttachmentCount;
}


RenderTarget::Desc::Desc()
{
    mColorAttachmentDescList.resize(getMaxColorAttachmentCount());
}


RenderTarget::Desc& RenderTarget::Desc::setColorAttachment(uint32_t index, const PixelFormat& format)
{
    mColorAttachmentDescList[index] = AttachmentDesc{format};
    return *this;
}


RenderTarget::Desc& RenderTarget::Desc::setDepthStencilAttachment(const PixelFormat& format)
{
    mDepthStencilAttachmentDesc = AttachmentDesc{format};
    return *this;
}


PixelFormat RenderTarget::Desc::getColorAttachmentFormat(uint32_t index) const
{
    if (index >= mColorAttachmentDescList.size()) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                                   "color attachment index [{}] is greater than the max attachment number [{}]",
                                   index, getMaxColorAttachmentCount());
    }
    return mColorAttachmentDescList[index].pixelFormat;
}


PixelFormat RenderTarget::Desc::getDepthStencilFormat() const
{
    return mDepthStencilAttachmentDesc.pixelFormat;
}




RenderTarget::RenderTarget(const std::string& name)
    : HardwareResource(name)
    , mRenderTextureMap()
    , mDepthTexture()
    , mClearColor(DEFAULT_CLEAR_COLOR_VALUE)
    , mClearDepthValue(DEFAULT_CLEAR_DEPTH_VALUE)
{
    SYRINX_ENSURE(mRenderTextureMap.empty());
    SYRINX_ENSURE(!mDepthTexture);
}


void RenderTarget::addRenderTexture(RenderTarget::RenderTextureBindingPoint bindingPoint, const RenderTexture& renderTexture)
{
    SYRINX_EXPECT(!isCreated());
    SYRINX_EXPECT(!renderTexture.isDepthTexture());

    auto iter = mRenderTextureMap.find(bindingPoint);
    if (iter != std::end(mRenderTextureMap)) {
        SYRINX_INFO_FMT("replace the render texture [{}] in binding point [{}] by render target [{}]",
                        iter->second.getName(),
                        bindingPoint,
                        renderTexture.getName());
    }
    mRenderTextureMap[bindingPoint] = renderTexture;
    SYRINX_ENSURE(mRenderTextureMap.find(bindingPoint) != std::end(mRenderTextureMap));
}


const RenderTexture* RenderTarget::getColorAttachment(RenderTarget::RenderTextureBindingPoint bindingPoint) const
{
    const auto iter = mRenderTextureMap.find(bindingPoint);
    if (iter == std::end(mRenderTextureMap)) {
        return nullptr;
    }
    return &(iter->second);
}


void RenderTarget::addDepthTexture(const DepthTexture& depthTexture)
{
    SYRINX_EXPECT(!isCreated());
    if (mDepthTexture) {
        SYRINX_DEBUG_FMT("change depth texture [{}] to [{}] for render target [{}]", mDepthTexture.getName(), depthTexture.getName(), getName());
    }
    mDepthTexture = depthTexture;
    SYRINX_ENSURE(mDepthTexture);
}


bool RenderTarget::create()
{
    SYRINX_EXPECT(!isCreated());
    SYRINX_EXPECT(isValidToCreate());

    GLuint handle = 0;
    glCreateFramebuffers(1, &handle);
    setHandle(handle);

    std::vector<GLenum> drawBufferList;
    for (auto i = 0; i < mRenderTextureMap.size(); ++ i) {
        auto bindingPoint = static_cast<RenderTextureBindingPoint>(i);
        auto renderTexture = mRenderTextureMap[bindingPoint];
        glNamedFramebufferTexture(getHandle(), GL_COLOR_ATTACHMENT0 + bindingPoint, renderTexture.getHandle(), 0);
        drawBufferList.push_back(GL_COLOR_ATTACHMENT0 + bindingPoint);
        SYRINX_INFO_FMT("render texture [{}] is bounded to render target [{}] in binding point [{}]", renderTexture.getName(), getName(), bindingPoint);
    }
    glNamedFramebufferDrawBuffers(getHandle(), static_cast<GLsizei>(mRenderTextureMap.size()), drawBufferList.data());

    glNamedFramebufferTexture(getHandle(), GL_DEPTH_ATTACHMENT, mDepthTexture.getHandle(), 0);
    SYRINX_INFO_FMT("depth texture [{}] is bounded to render target [{}]", mDepthTexture.getName(), getName());

    SYRINX_ENSURE(isValidStatus());
    SYRINX_ENSURE(isCreated());
    return true;
}


void RenderTarget::setClearColorValue(const Color& color)
{
    mClearColor = color;
}


void RenderTarget::setClearDepthValue(float depthValue)
{
    SYRINX_EXPECT(depthValue >= 0.0 && depthValue <= 1.0);
    mClearDepthValue = depthValue;
}


void RenderTarget::clearRenderTexture()
{
    SYRINX_EXPECT(isCreated());
    for (const auto& [bindingPoint, renderTexture] : mRenderTextureMap) {
        glClearNamedFramebufferfv(getHandle(), GL_COLOR, bindingPoint, mClearColor);
    }
}


void RenderTarget::clearDepthTexture()
{
    SYRINX_EXPECT(isCreated());
    glClearNamedFramebufferfv(getHandle(), GL_DEPTH, 0, &mClearDepthValue);
}


bool RenderTarget::isValidStatus()
{
    SYRINX_EXPECT(isCreated());
    if (glCheckNamedFramebufferStatus(getHandle(), GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        SYRINX_ERROR_FMT("render target [{}] is not valid", getName());
        return false;
    }
    return true;
}


bool RenderTarget::isValidToCreate() const
{
    if (!mDepthTexture) {
        SYRINX_ERROR_FMT("render target [{}] doesn't have depth texture", getName());
        return false;
    }

    if (mRenderTextureMap.empty()) {
        SYRINX_ERROR_FMT("render target [{}] doesn't have render texture", getName());
        return false;
    }
    return true;
}

} // namespace Syrinx