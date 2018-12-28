#pragma once
#include <cstdint>
#include "Serializer/SyrinxSerializerCommon.h"
#include "Streaming/SyrinxDataStream.h"

namespace Syrinx {

class Deserializer {
public:
    Deserializer();
    virtual ~Deserializer() = default;

protected:
    virtual void readCustomHeader() noexcept(false) = 0;
    virtual void deserializeData() noexcept(false) = 0;
    virtual void clear() = 0;
    virtual bool isClean() const = 0;

    void deserializeFileHeader() noexcept(false);
    void readData(void *data, size_t elemSize, size_t cnt);
    template <typename T> inline void readData(T *data, size_t cnt);
    void readUInt8s(uint8_t *data, size_t cnt);
    void readUInt16s(uint16_t *data, size_t cnt);
    void readUInt32s(uint32_t *data, size_t cnt);
    void readUInt64s(uint64_t *data, size_t cnt);
    void readFloats(float *data, size_t cnt);
    void readFloats(double *data, size_t cnt);

    bool readBool();
    uint8_t readUInt8();
    uint16_t readUInt16();
    uint32_t readUInt32();
    uint64_t readUInt64();
    std::string readString();

protected:
    DataStream *mDataStream;
    std::string mVersion;
    Endian mEndian;
};

} // namespace Syrinx