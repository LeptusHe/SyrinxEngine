#pragma once
#include <cassert>

#define SYRINX_ASSERT(expression) assert(expression)
#define SYRINX_EXPECT(expression) assert(expression)
#define SYRINX_ENSURE(expression) assert(expression)


#ifdef _MSC_VER
#define SHOULD_NOT_GET_HERE() __assume(0)
#else
#define SHOULD_NOT_GET_HERE() __builtin_unreachable()
#endif