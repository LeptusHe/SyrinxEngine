#pragma once
#include <better-enums/enum.h>
#ifndef GLEW_STATIC
#define GLEW_STATIC
#endif // GLEW_STATIC
#include <GL/glew.h>

namespace Syrinx {

BETTER_ENUM(HardwareResourceState, uint8_t, Uncreated, Created);


class HardwareResourceHandle {
public:
   static constexpr GLuint INVALID_RESOURCE_HANDLE_VALUE = 0;

public:
    void setHandleValue(GLuint value);
    GLuint getHandleValue() const;
    GLuint& fetchHandleValue();

private:
    GLuint mHandleValue = INVALID_RESOURCE_HANDLE_VALUE;
};


class HardwareResource {
public:
    using ResourceHandle = GLuint;

public:
    explicit HardwareResource(const std::string& name);
    virtual ~HardwareResource() = default;

    virtual bool create() = 0;
    const std::string& getName() const;
    HardwareResourceState getState() const;
    ResourceHandle getHandle() const;
    ResourceHandle& fetchHandle();
    bool isCreated() const;

protected:
    virtual bool isValidToCreate() const = 0;
    void setHandle(ResourceHandle handle);
    void setState(HardwareResourceState state);

private:
    std::string mName;
    HardwareResourceHandle mHandle;
    HardwareResourceState mState;
};

} // namespace Syrinx
