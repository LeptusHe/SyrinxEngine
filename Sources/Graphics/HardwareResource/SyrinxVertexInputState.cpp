#include <algorithm>
#include "HardwareResource/SyrinxVertexInputState.h"
#include <Common/SyrinxConfig.h>
#include <Logging/SyrinxLogManager.h>
#include "HardwareResource/SyrinxConstantTranslator.h"

namespace Syrinx {

VertexBufferLayoutDesc::VertexBufferLayoutDesc(VertexBufferLayoutDesc&& rhs) noexcept
    : mBindingPoint(rhs.mBindingPoint)
    , mVertexAttributeDescList(std::move(rhs.mVertexAttributeDescList))
{

}


VertexBufferLayoutDesc& VertexBufferLayoutDesc::operator=(VertexBufferLayoutDesc&& rhs) noexcept
{
    if (&rhs == this) {
        return *this;
    }

    mBindingPoint = rhs.mBindingPoint;
    mVertexAttributeDescList = std::move(rhs.mVertexAttributeDescList);
    return *this;
}


void VertexBufferLayoutDesc::setBindingPoint(VertexBufferBindingPoint bindingPoint)
{
    mBindingPoint = bindingPoint;
}


void VertexBufferLayoutDesc::addVertexAttributeDesc(const VertexAttributeDescription& vertexAttributeDesc)
{
    if (vertexAttributeDesc.getBindingPoint() != mBindingPoint) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
            "fail to add vertex attribute description [semantic = {}]", vertexAttributeDesc.getSemantic()._to_string());
    }

    auto iter = mVertexAttributeDescList.begin();
    size_t nextEndPos = 0;
    while (iter != mVertexAttributeDescList.end()) {
        auto dataType = iter->getDataType();
        auto dataOffset = iter->getDataOffset();
        auto dataSize = VertexAttributeDescription::getByteSizeForDataType(dataType);

        nextEndPos = dataOffset + dataSize;
        if (nextEndPos > vertexAttributeDesc.getDataOffset()) {
            break;
        }
        iter++;
    }
    if (iter != mVertexAttributeDescList.end()) {
        auto dataType = vertexAttributeDesc.getDataType();
        auto dataOffset = vertexAttributeDesc.getDataOffset();
        auto dataSize = VertexAttributeDescription::getByteSizeForDataType(dataType);
        if (dataOffset + dataSize > iter->getDataOffset()) {
            SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                "fail to add vertex attribute [semantic = {}] into vertex buffer layout description", vertexAttributeDesc.getSemantic()._to_string());
        }
    }
    mVertexAttributeDescList.insert(iter, vertexAttributeDesc);
    SYRINX_ENSURE(!mVertexAttributeDescList.empty());
}


VertexBufferBindingPoint VertexBufferLayoutDesc::getBindingPoint() const
{
    return mBindingPoint;
}


size_t VertexBufferLayoutDesc::getStride() const
{
    if (mVertexAttributeDescList.empty()) {
        return 0;
    }
    auto iter = mVertexAttributeDescList.back();
    auto dataOffset = iter.getDataOffset();
    auto dataType = iter.getDataType();
    auto dataSize = VertexAttributeDescription::getByteSizeForDataType(dataType);
    return dataOffset + dataSize;
}


size_t VertexBufferLayoutDesc::getVertexAttributeCount() const
{
    return mVertexAttributeDescList.size();
}


const std::list<VertexAttributeDescription>& VertexBufferLayoutDesc::getVertexAttributeDescriptionList() const
{
    return mVertexAttributeDescList;
}


VertexAttributeLayoutDesc::VertexAttributeLayoutDesc(VertexAttributeLayoutDesc&& rhs) noexcept
    : mVertexBufferLayoutDescList(std::move(rhs.mVertexBufferLayoutDescList))
{

}


VertexAttributeLayoutDesc& VertexAttributeLayoutDesc::operator=(VertexAttributeLayoutDesc&& rhs) noexcept
{
    mVertexBufferLayoutDescList = std::move(rhs.mVertexBufferLayoutDescList);
    return *this;
}


void VertexAttributeLayoutDesc::addVertexAttributeDesc(const VertexAttributeDescription& vertexAttributeDescription)
{
    auto bindingPoint = vertexAttributeDescription.getBindingPoint();
    if (bindingPoint + 1 > mVertexBufferLayoutDescList.size()) {
        auto oldSize = mVertexBufferLayoutDescList.size();
        mVertexBufferLayoutDescList.resize(bindingPoint + 1);
        for (int i = oldSize; i < mVertexBufferLayoutDescList.size(); ++i) {
            mVertexBufferLayoutDescList[i].setBindingPoint(i);
        }
    }
    mVertexBufferLayoutDescList[bindingPoint].addVertexAttributeDesc(vertexAttributeDescription);
}


const std::vector<VertexBufferLayoutDesc>& VertexAttributeLayoutDesc::getVertexBufferLayoutDescList() const
{
    return mVertexBufferLayoutDescList;
}


size_t VertexAttributeLayoutDesc::getVertexAttributeCount() const
{
    size_t count = 0;
    for (const auto& vertexBufferLayout : mVertexBufferLayoutDescList) {
        count += vertexBufferLayout.getVertexAttributeCount();
    }
    return count;
}




VertexInputState::VertexInputState(const std::string& name)
    : HardwareResource(name)
    , mVertexAttributeLayoutDesc()
    , mVertexBufferList()
    , mIndexBuffer()
    , mBufferStrideList()
{
    SYRINX_ENSURE(!mIndexBuffer);
}


VertexInputState::~VertexInputState()
{
    auto handle = getHandle();
    glDeleteVertexArrays(1, &handle);
}


void VertexInputState::setVertexAttributeLayoutDesc(VertexAttributeLayoutDesc&& vertexAttributeLayoutDesc)
{
    mVertexAttributeLayoutDesc = std::move(vertexAttributeLayoutDesc);
}


void VertexInputState::setVertexBuffer(const VertexBufferBindingPoint& bindingPoint, const HardwareVertexBuffer *vertexBuffer)
{
    SYRINX_EXPECT(vertexBuffer);
    if (bindingPoint >= mVertexBufferList.size()) {
        mVertexBufferList.resize(bindingPoint + 1, nullptr);
    }
    mVertexBufferList[bindingPoint] = vertexBuffer;
}


void VertexInputState::setIndexBuffer(const HardwareIndexBuffer *indexBuffer)
{
    SYRINX_EXPECT(indexBuffer);
    mIndexBuffer = indexBuffer;
}


const HardwareVertexBuffer* VertexInputState::getVertexBuffer(const VertexBufferBindingPoint& bindingPoint) const
{
    if (bindingPoint >= mVertexBufferList.size()) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
            "fail to get vertex buffer in binding point [{}] in vertex input state [{}] because the binding point is invalid", getName(), bindingPoint);
    }
    return mVertexBufferList[bindingPoint];
}


const HardwareIndexBuffer* VertexInputState::getIndexBuffer() const
{
    return mIndexBuffer;
}


bool VertexInputState::create()
{
    SYRINX_EXPECT(!isCreated());
    SYRINX_EXPECT(isValidToCreate());

    GLuint handle = 0;
    glCreateVertexArrays(1, &handle);
    setHandle(handle);
    SYRINX_ENSURE(isCreated());
    return true;
}


void VertexInputState::setup()
{
    if (!mIndexBuffer) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidState, "fail to set up vertex input state [{}] because there is no index buffer", getName());
    }

    auto numVertex = mIndexBuffer->getNumIndexes();

    const auto& vertexBufferLayoutDescList = mVertexAttributeLayoutDesc.getVertexBufferLayoutDescList();
    for (const auto& vertexBufferLayoutDesc : vertexBufferLayoutDescList) {
        auto bindingPoint = vertexBufferLayoutDesc.getBindingPoint();
        auto vertexBuffer = mVertexBufferList[bindingPoint];

        if (vertexBufferLayoutDesc.getVertexAttributeCount() == 0) {
            if (vertexBuffer) {
                SYRINX_WARN_FMT("vertex input state [{}] has vertex buffer bound to [{}] but there is no vertex attribute read data from it", getName(), bindingPoint);
            }
        } else {
            if (!vertexBuffer) {
                SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidState,
                    "vertex input state [{}] has vertex attribute bound to binding point [{}] but no vertex buffer bound to the binding point",
                    getName(), bindingPoint);
            }

            if (vertexBufferLayoutDesc.getStride() != vertexBuffer->getVertexSizeInBytes()) {
                SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidState,
                    "fail to set up vertex input state [{}] because the stride of vertex buffer layout description [{}] is not the same as the vertex size [{}] of vertex buffer [{}]",
                    getName(), vertexBufferLayoutDesc.getStride(), vertexBuffer->getVertexSizeInBytes(), vertexBuffer->getBuffer().getName());
            }

            if (false && numVertex != vertexBuffer->getVertexNumber()) {
                SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidState,
                    "fail to set up vertex input state [{}] because the vertex number of index buffer [{}] is not the same as vertex buffer [{}]",
                    getName(), mIndexBuffer->getBuffer().getName(), vertexBuffer->getBuffer().getName());
            }
        }
    }

    auto handle = getHandle();
    for (const auto& vertexBufferLayoutDesc : vertexBufferLayoutDescList) {
        auto bindingPoint = vertexBufferLayoutDesc.getBindingPoint();
        auto vertexBuffer = mVertexBufferList[bindingPoint];
        SYRINX_ASSERT(vertexBuffer);

        for (const auto& vertexAttributeDesc : vertexBufferLayoutDesc.getVertexAttributeDescriptionList()) {
            SYRINX_ASSERT(vertexAttributeDesc.getBindingPoint() == bindingPoint);
            auto dataType = vertexAttributeDesc.getDataType();
            auto dataOffset = vertexAttributeDesc.getDataOffset();
            auto dataSize = VertexAttributeDescription::getByteSizeForDataType(dataType);
            auto location = vertexAttributeDesc.getLocation();
            const auto [numValue, valueType] = ConstantTranslator::getOpenGLValueType(dataType);

            glEnableVertexArrayAttrib(handle, location);
            glVertexArrayAttribFormat(handle, location, numValue, valueType, GL_FALSE, dataOffset);
            glVertexArrayAttribBinding(handle, location, bindingPoint);
        }
        glVertexArrayVertexBuffer(handle, bindingPoint, vertexBuffer->getHandle(), 0, vertexBufferLayoutDesc.getStride());
    }
    glVertexArrayElementBuffer(handle, mIndexBuffer->getHandle());
}


bool VertexInputState::isValidToCreate() const
{
    return true;
}

} // namespace Syrinx
