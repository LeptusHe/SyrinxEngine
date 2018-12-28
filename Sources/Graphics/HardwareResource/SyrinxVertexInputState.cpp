#include "HardwareResource/SyrinxVertexInputState.h"
#include <Common/SyrinxConfig.h>
#include <Logging/SyrinxLogManager.h>
#include "HardwareResource/SyrinxConstantTranslator.h"

namespace Syrinx {

VertexInputState::VertexInputState(const std::string& name)
    : HardwareResource(name)
    , mHardwareIndexBuffer()
{
    SYRINX_ENSURE(!mHardwareIndexBuffer);
}


void VertexInputState::addVertexAttributeDescription(const VertexAttributeDescription& vertexAttributeDescription)
{
    auto bindingPoint = vertexAttributeDescription.getBindingPoint();
    if (mVertexAttributeDescriptionMap.find(bindingPoint) != std::end(mVertexAttributeDescriptionMap)) {
        auto& preVertexAttributeDescription = mVertexAttributeDescriptionMap.find(bindingPoint)->second;
        auto preSemantic = preVertexAttributeDescription.getSemantic()._to_string();
        auto preDataType = preVertexAttributeDescription.getDataType()._to_string();
        auto curSemantic = vertexAttributeDescription.getSemantic()._to_string();
        auto curDataType = vertexAttributeDescription.getDataType()._to_string();
        SYRINX_DEBUG_FMT("change vertex attribute binding point [{}], before=[{}, {}], after=[{}, {}]", bindingPoint, preSemantic, preDataType, curSemantic, curDataType);
    }
    mVertexAttributeDescriptionMap.insert({bindingPoint, vertexAttributeDescription});
}


void VertexInputState::addVertexDataDescription(const VertexDataDescription& vertexInputDataDescription)
{
    auto bindingPoint = vertexInputDataDescription.getVertexBufferBindingPoint();
    if (mVertexDataDescriptionMap.find(bindingPoint) != std::end(mVertexDataDescriptionMap)) {
        SYRINX_DEBUG_FMT("change vertex buffer binding point [{}]", bindingPoint);
    }
    mVertexDataDescriptionMap.insert({bindingPoint, vertexInputDataDescription});
}


void VertexInputState::addIndexBuffer(const HardwareIndexBuffer *indexBuffer)
{
    SYRINX_EXPECT(indexBuffer);
    mHardwareIndexBuffer = indexBuffer;
    SYRINX_ENSURE(mHardwareIndexBuffer);
    SYRINX_ENSURE(mHardwareIndexBuffer == indexBuffer);
}


const HardwareIndexBuffer& VertexInputState::getIndexBuffer() const
{
    SYRINX_EXPECT(mHardwareIndexBuffer);
    return *mHardwareIndexBuffer;
}


const VertexInputState::VertexDataDescriptionMap& VertexInputState::getVertexDataDescriptionMap() const
{
    return mVertexDataDescriptionMap;
}


const VertexInputState::VertexAttributeDescriptionMap& VertexInputState::getVertexAttributeDescriptionMap() const
{
    return mVertexAttributeDescriptionMap;
}


const VertexInputState::VertexDataDescriptionMap::const_iterator VertexInputState::getVertexDataDescription(VertexBufferBindingPoint bindingPoint) const
{
    return mVertexDataDescriptionMap.find(bindingPoint);
}


const VertexInputState::VertexAttributeDescriptionMap::const_iterator VertexInputState::getVertexAttributeDescription(VertexAttributeBindingPoint bindingPoint) const
{
    return mVertexAttributeDescriptionMap.find(bindingPoint);
}


bool VertexInputState::create()
{
    SYRINX_EXPECT(!isCreated());
    SYRINX_EXPECT(isValidToCreate());

    GLuint handle = 0;
    glCreateVertexArrays(1, &handle);

    for (const auto& [bindingPoint, vertexAttributeDescription] : mVertexAttributeDescriptionMap) {
        const auto [valueNumber, valueType] = ConstantTranslator::getOpenGLValueType(vertexAttributeDescription.getDataType());
        auto vertexDataDescription = getVertexDataDescription(bindingPoint)->second;
        auto vertexBuffer = vertexDataDescription.getVertexBuffer();
        GLuint bufferHandle = vertexBuffer->getHandle();
        size_t offset = vertexDataDescription.getOffsetOfFirstElement();
        size_t stride = vertexDataDescription.getStrideBetweenElements();
        glEnableVertexArrayAttrib(handle, bindingPoint);
        glVertexArrayAttribFormat(handle, bindingPoint, valueNumber, valueType, GL_FALSE, 0);
        glVertexArrayVertexBuffer(handle, bindingPoint, bufferHandle, offset, stride);
        glVertexArrayAttribBinding(handle, bindingPoint, bindingPoint);
    }
    glVertexArrayElementBuffer(handle, mHardwareIndexBuffer->getHandle());
    setHandle(handle);
    SYRINX_ENSURE(isCreated());
    return true;
}


bool VertexInputState::isValidToCreate() const
{
    if (mVertexAttributeDescriptionMap.empty() || mVertexDataDescriptionMap.empty())
        return false;

    if (!mHardwareIndexBuffer)
        return false;

    bool isValid = true;
    for (const auto& [bindingPoint, vertexAttribute] : mVertexAttributeDescriptionMap) {
        auto vertexSemantic = vertexAttribute.getSemantic()._to_string();
        const auto iter = mVertexDataDescriptionMap.find(bindingPoint);
        if (iter == std::cend(mVertexDataDescriptionMap)) {
            SYRINX_DEBUG_FMT("no data source for vertex attribute [binding point={}, semantic={}]", bindingPoint, vertexSemantic);
            isValid = false;
        } else {
            const auto vertexBuffer = iter->second.getVertexBuffer();
            if (!vertexBuffer->isCreated()) {
                SYRINX_DEBUG_FMT("the vertex buffer for vertex attribute [binding point={}, semantic={}] was not created", bindingPoint, vertexSemantic);
                isValid = false;
            }
        }
    }
    return isValid;
}

} // namespace Syrinx
