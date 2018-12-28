#pragma once

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    #define SYRINX_OS_WINDOWS
#elif defined(__linux) || defined(__linux__)
    #define SYRINX_OS_LINUX
#else
    #error unsupport platform
#endif