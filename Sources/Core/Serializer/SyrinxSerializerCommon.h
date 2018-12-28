#pragma once
#include <cstdint>

namespace Syrinx {

constexpr uint32_t SERIALIZER_HEADER_CHECKER = 0x1309262;
constexpr char SerializerVersionInfo[] = "[SyrinxSerializer_v1.00]";

enum class Endian {
    LITTLE = 0,
    BIG = 1,
    COUNT = 2,
    DEFAULT = LITTLE
};

} // namespace Syrinx