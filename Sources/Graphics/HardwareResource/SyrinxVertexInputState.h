#pragma once
#include <list>
#include <vector>
#include "SyrinxHardwareResource.h"
#include "SyrinxHardwareIndexBuffer.h"
#include "SyrinxVertexDataDescription.h"
#include "SyrinxVertexAttributeDescription.h"

namespace Syrinx {

class VertexBufferLayoutDesc {
public:
    VertexBufferLayoutDesc() = default;
    VertexBufferLayoutDesc(VertexBufferLayoutDesc&& rhs) noexcept;
    VertexBufferLayoutDesc& operator=(VertexBufferLayoutDesc&& rhs) noexcept;

    void setBindingPoint(VertexBufferBindingPoint bindingPoint);
    void addVertexAttributeDesc(const VertexAttributeDescription& vertexAttributeDesc);
    VertexBufferBindingPoint getBindingPoint() const;
    size_t getStride() const;
    size_t getVertexAttributeCount() const;
    const std::list<VertexAttributeDescription>& getVertexAttributeDescriptionList() const;

private:
    VertexBufferBindingPoint mBindingPoint = 0;
    std::list<VertexAttributeDescription> mVertexAttributeDescList;
};




class VertexAttributeLayoutDesc {
public:
    VertexAttributeLayoutDesc() = default;
    VertexAttributeLayoutDesc(VertexAttributeLayoutDesc&& rhs) noexcept;
    VertexAttributeLayoutDesc& operator=(VertexAttributeLayoutDesc&& rhs) noexcept;
    void addVertexAttributeDesc(const VertexAttributeDescription& vertexAttributeDescription);
    const std::vector<VertexBufferLayoutDesc>& getVertexBufferLayoutDescList() const;
    size_t getVertexAttributeCount() const;

private:
    std::vector<VertexBufferLayoutDesc> mVertexBufferLayoutDescList;
};




class VertexInputState : public HardwareResource {
public:
    explicit VertexInputState(const std::string& name);
    ~VertexInputState() override;

    void setVertexAttributeLayoutDesc(VertexAttributeLayoutDesc&& vertexAttributeLayoutDesc);
    void setVertexBuffer(const VertexBufferBindingPoint& bindingPoint, const HardwareVertexBuffer *vertexBuffer);
    void setIndexBuffer(const HardwareIndexBuffer *indexBuffer);
    const HardwareVertexBuffer *getVertexBuffer(const VertexBufferBindingPoint& bindingPoint) const;
    const HardwareIndexBuffer *getIndexBuffer() const;
    bool create() override;
    void setup();

protected:
    bool isValidToCreate() const override;

private:
    VertexAttributeLayoutDesc mVertexAttributeLayoutDesc;
    std::vector<const HardwareVertexBuffer*> mVertexBufferList;
    const HardwareIndexBuffer *mIndexBuffer;
    std::vector<uint32_t> mBufferStrideList;
};

} // namespace Syrinx
