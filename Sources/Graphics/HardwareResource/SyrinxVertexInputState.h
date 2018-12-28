#pragma once
#include <map>
#include "HardwareResource/SyrinxHardwareResource.h"
#include "HardwareResource/SyrinxHardwareIndexBuffer.h"
#include "HardwareResource/SyrinxVertexDataDescription.h"
#include "HardwareResource/SyrinxVertexAttributeDescription.h"

namespace Syrinx {

class VertexInputState : public HardwareResource {
public:
    using VertexAttributeDescriptionMap = std::map<VertexAttributeBindingPoint, VertexAttributeDescription>;
    using VertexDataDescriptionMap = std::map<VertexBufferBindingPoint, VertexDataDescription>;

public:
    explicit VertexInputState(const std::string& name);
    ~VertexInputState() override = default;

    void addVertexAttributeDescription(const VertexAttributeDescription& vertexAttributeDescription);
    void addVertexDataDescription(const VertexDataDescription& vertexInputDataDescription);
    void addIndexBuffer(const HardwareIndexBuffer *indexBuffer);
    const HardwareIndexBuffer& getIndexBuffer() const;
    const VertexDataDescriptionMap& getVertexDataDescriptionMap() const;
    const VertexAttributeDescriptionMap& getVertexAttributeDescriptionMap() const;
    const VertexDataDescriptionMap::const_iterator getVertexDataDescription(VertexBufferBindingPoint bindingPoint) const;
    const VertexAttributeDescriptionMap::const_iterator getVertexAttributeDescription(VertexAttributeBindingPoint bindingPoint) const;
    bool create() override;

protected:
    bool isValidToCreate() const override;

private:
    VertexAttributeDescriptionMap mVertexAttributeDescriptionMap;
    VertexDataDescriptionMap mVertexDataDescriptionMap;
    const HardwareIndexBuffer *mHardwareIndexBuffer;
};

} // namespace Syrinx
