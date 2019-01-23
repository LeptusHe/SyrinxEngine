#include "RenderPipeline/SyrinxEngineSetting.h"
#include <Common/SyrinxAssert.h>
#include <iostream>

namespace Syrinx {

void DefaultDebugHandler(GLenum source,
                         GLenum type,
                         GLuint id,
                         GLenum severity,
                         GLsizei length,
                         const GLchar *message,
                         const void *userParam)
{
    if (id == 131169 || id == 131185 || id == 131218 || id == 131204)
        return;

    if (severity == GL_DEBUG_SEVERITY_LOW)
        return;

    //TODO: format debug meesage
    std::cout << "---------------" << std::endl;
    std::cout << "Debug message (" << id << "): " << message << std::endl;

    switch (source) {
        case GL_DEBUG_SOURCE_API:
            std::cout << "Source: API";
            break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
            std::cout << "Source: Window System";
            break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER:
            std::cout << "Source: Shader Compiler";
            break;
        case GL_DEBUG_SOURCE_THIRD_PARTY:
            std::cout << "Source: Third Party";
            break;
        case GL_DEBUG_SOURCE_APPLICATION:
            std::cout << "Source: Application";
            break;
        case GL_DEBUG_SOURCE_OTHER:
            std::cout << "Source: Other";
            break;
        default:
            SYRINX_ASSERT(false);
    };
    std::cout << std::endl;

    switch (type) {
        case GL_DEBUG_TYPE_ERROR:
            std::cout << "Type: Error";
            break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
            std::cout << "Type: Deprecated Behaviour";
            break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
            std::cout << "Type: Undefined Behaviour";
            break;
        case GL_DEBUG_TYPE_PORTABILITY:
            std::cout << "Type: Portability";
            break;
        case GL_DEBUG_TYPE_PERFORMANCE:
            std::cout << "Type: Performance";
            break;
        case GL_DEBUG_TYPE_MARKER:
            std::cout << "Type: Marker";
            break;
        case GL_DEBUG_TYPE_PUSH_GROUP:
            std::cout << "Type: Push Group";
            break;
        case GL_DEBUG_TYPE_POP_GROUP:
            std::cout << "Type: Pop Group";
            break;
        case GL_DEBUG_TYPE_OTHER:
            std::cout << "Type: Other";
            break;
        default:
            SYRINX_ASSERT(false);
    };
    std::cout << std::endl;

    switch (severity) {
        case GL_DEBUG_SEVERITY_HIGH:
            std::cout << "Severity: high";
            break;
        case GL_DEBUG_SEVERITY_MEDIUM:
            std::cout << "Severity: medium";
            break;
        case GL_DEBUG_SEVERITY_LOW:
            std::cout << "Severity: low";
            break;
        case GL_DEBUG_SEVERITY_NOTIFICATION:
            std::cout << "Severity: notification";
            break;
        default:
            SYRINX_ASSERT(false);
    };
    std::cout << std::endl << std::endl;
}


EngineSetting::EngineSetting() : EngineSetting(4, 5, DefaultDebugHandler)
{

}


EngineSetting::EngineSetting(unsigned MajorVersion,
                             unsigned MinorVersion,
                             DebugMessageHandler debugMessageHandler)
    : mMajorVersion(MajorVersion)
    , mMinorVersion(MinorVersion)
    , mDebugMessageHandler(debugMessageHandler)
{
    SYRINX_ENSURE(mMajorVersion == MajorVersion);
    SYRINX_ENSURE(mMinorVersion == MinorVersion);
    SYRINX_ENSURE(mDebugMessageHandler == debugMessageHandler);
    SYRINX_ENSURE(isVersionValid());
}


void EngineSetting::setVersionNumber(unsigned majorVersionNumber, unsigned minorVersionNumber)
{
    mMajorVersion = majorVersionNumber;
    mMinorVersion = minorVersionNumber;
    SYRINX_ENSURE(isVersionValid());
    SYRINX_ENSURE(mMajorVersion == majorVersionNumber && mMinorVersion == minorVersionNumber);
}


void EngineSetting::setDebugMessageHandler(DebugMessageHandler debugMessageHandler)
{
    mDebugMessageHandler = debugMessageHandler;
    SYRINX_ENSURE(mDebugMessageHandler == debugMessageHandler);
}


unsigned EngineSetting::getMajorVersionNumber() const
{
    return mMajorVersion;
}


unsigned EngineSetting::getMinorVersionNumber() const
{
    return mMinorVersion;
}


DebugMessageHandler EngineSetting::getDebugMessageHandler() const
{
    return mDebugMessageHandler;
}


bool EngineSetting::isVersionValid() const
{
    return mMajorVersion == 4 && mMinorVersion >= 5;
}

} // namespace Syrinx