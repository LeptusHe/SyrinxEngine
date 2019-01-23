#pragma once

#if defined(__CYGWIN__)
    #define SYRINX_OS_CYGWIN
#elif defined(__MINGW32__) || defined(__MINGW64__)
    #define SYRINX_OS_MINGW32
#elif defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    #define SYRINX_OS_WINDOWS
#elif defined(__linux) || defined(__linux__)
    #define SYRINX_OS_LINUX
#else
    #error unsupport platform
#endif