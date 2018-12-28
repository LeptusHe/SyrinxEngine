#pragma once
#include "Streaming/SyrinxDataStream.h"
#include "Serializer/SyrinxSerializerCommon.h"

namespace Syrinx {

class Serializer {
public:
    Serializer();
    virtual ~Serializer() = default;

protected:
    virtual void writeCustomHeader() = 0;
    virtual void serializeData() = 0;
    virtual void clear() = 0;
    virtual bool isClean() const = 0;

    void serializeFileHeader();
    void writeData(const void *data, size_t elemSize, size_t cnt);
    template <typename T> inline void writeData(T *data, size_t cnt);
    void writeUInt8s(const uint8_t *data, size_t cnt);
    void writeUInt16s(const uint16_t *data, size_t cnt);
    void writeUInt32s(const uint32_t *data, size_t cnt);
    void writeUInt64s(const uint64_t *data, size_t cnt);
    void writeFloats(const float *data, size_t cnt);
    void writeFloats(const double *data, size_t cnt);

    void writeBool(bool value) { writeUInt8(static_cast<uint8_t>(value)); }
    void writeUInt8(uint8_t value) { writeUInt8s(&value, 1); }
    void writeUInt16(uint16_t value) { writeUInt16s(&value, 1); }
    void writeUInt32(uint32_t value) { writeUInt32s(&value, 1); }
    void writeUInt64(uint64_t value) { writeUInt64s(&value, 1); }
    void writeString(const std::string& str);

protected:
    DataStream *mDataStream;
    std::string mVersion;
    Endian mEndian;
};

} // namespace Syrinx