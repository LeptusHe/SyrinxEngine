#include "SyrinxDepthTexture.h"
#include <Common/SyrinxAssert.h>
#include <Exception/SyrinxException.h>

namespace Syrinx {

DepthTexture::DepthTexture()
    : mName()
    , mDepthTexture()
{
    SYRINX_EXPECT(mName.empty());
    SYRINX_EXPECT(!mDepthTexture);
}


DepthTexture::DepthTexture(const std::string& name, const RenderTexture& depthTexture)
    : mName(name)
    , mDepthTexture(depthTexture)
{
    SYRINX_ENSURE(!mName.empty());
    SYRINX_ENSURE(mDepthTexture);
    SYRINX_ENSURE(mDepthTexture.isDepthTexture());
}


DepthTexture& DepthTexture::operator=(const DepthTexture& rhs)
{
    if (this == &rhs) {
        return *this;
    }
    mName = rhs.mName;
    mDepthTexture = rhs.mDepthTexture;
    return *this;
}


DepthTexture::operator bool() const
{
    return static_cast<bool>(mDepthTexture);
}


HardwareResource::ResourceHandle DepthTexture::getHandle() const
{
    SYRINX_EXPECT(mDepthTexture);
    return mDepthTexture.getHandle();
}


const std::string& DepthTexture::getName() const
{
    return mName;
}


void DepthTexture::setName(const std::string& name)
{
    SYRINX_EXPECT(mName.empty());
    mName = name;
    SYRINX_ENSURE(!mName.empty());
}


void DepthTexture::setRenderTexture(const RenderTexture& renderTexture)
{
    SYRINX_EXPECT(renderTexture);
    SYRINX_EXPECT(!mDepthTexture);
    if (!renderTexture.isDepthTexture()) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                                   "fail to set render texture [{}] to depth texture [{}] because it's format can not be used for depth texture",
                                   renderTexture.getName(), getName());
    }
    mDepthTexture = renderTexture;
    SYRINX_ENSURE(mDepthTexture);
}

} // namespace Syrinx