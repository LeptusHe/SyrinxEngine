#pragma once
#include <string>

namespace Syrinx {

class RenderResource {
public:
    explicit RenderResource(const std::string& name);
    virtual ~RenderResource() = default;

    const std::string& getName() const;

private:
    std::string mName;
};

} // namespace Syrinx