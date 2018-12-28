#pragma once
#include <memory>
#ifndef GLEW_STATIC
#define GLEW_STATIC
#endif //GLEW_STATIC
#include <GL/glew.h>

namespace Syrinx {

extern void DefaultDebugHandler(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *message, const void *userParam);

using DebugMessageHandler = std::add_pointer<void(GLenum, GLenum, GLuint, GLenum, GLsizei, const GLchar *, const void *)>::type;


class EngineSetting {
public:
    EngineSetting();
    EngineSetting(unsigned MajorVersion, unsigned MinorVersion, DebugMessageHandler debugMessageHandler);
    ~EngineSetting() = default;

    void setVersionNumber(unsigned majorVersionNumber, unsigned minorVersionNumber);
    void setDebugMessageHandler(DebugMessageHandler debugMessageHandler);
    unsigned getMajorVersionNumber() const;
    unsigned getMinorVersionNumber() const;
    DebugMessageHandler getDebugMessageHandler() const;
    bool isVersionValid() const;

private:
    unsigned mMajorVersion;
    unsigned mMinorVersion;
    DebugMessageHandler mDebugMessageHandler;
};

} // namespace Syrinx
