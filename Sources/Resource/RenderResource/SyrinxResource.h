#pragma once
#include <string>

namespace Syrinx {

class Resource {
public:
    explicit Resource(const std::string& name);
    virtual ~Resource() = default;

    const std::string& getName() const;

private:
    std::string mName;
};

} // namespace Syrinx