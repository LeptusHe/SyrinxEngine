#pragma once
#include "Common/SyrinxPlatform.h"
#include "Common/SyrinxMacro.h"


#if defined(SYRINX_OS_WINDOWS) && !NDEBUG
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>


inline void installMemoryLeakDetector()
{
#if defined(DEBUG) | defined(_DEBUG) | !defined(NDEBUG)
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
}


#if defined(_DEBUG) | !defined(NDEBUG)
    #define SYRINX_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
    #define SYRINX_DELETE delete
    #define SYRINX_NEW_ARRAY(T, size) new T[size];
    #define SYRINX_DELETE_ARRAY(ptr)  delete[] ptr 
#else
    #define SYRINX_NEW new
    #define SYRINX_DELETE delete
    #define SYRINX_NEW_ARRAY(T, size) new T[size]
    #define SYRINX_DELETE_ARRAY(ptr)  delete[] ptr;
#endif

#endif