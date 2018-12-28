#include "Serializer/SyrinxSerializer.h"
#include "Common/SyrinxAssert.h"

namespace Syrinx {

Serializer::Serializer(): mDataStream(nullptr), mVersion(SerializerVersionInfo), mEndian(Endian::DEFAULT)
{
    SYRINX_ENSURE(!mDataStream && mEndian == Endian::DEFAULT);
}


void Serializer::serializeFileHeader()
{
    writeUInt32(SERIALIZER_HEADER_CHECKER);
    writeString(mVersion);
    writeUInt8(static_cast<uint8_t>(mEndian));
    writeCustomHeader();
}


void Serializer::writeData(const void *data, size_t elemSize, size_t cnt)
{
    SYRINX_EXPECT(data);
    mDataStream->write(data, elemSize * cnt);
}


template <typename T>
void Serializer::writeData(T *data, size_t cnt)
{
    SYRINX_EXPECT(mDataStream && mDataStream->isWriteable() && data);
    mDataStream->write(data, sizeof(T) * cnt);
}


void Serializer::writeUInt8s(const uint8_t *data, size_t cnt)
{
    SYRINX_EXPECT(data);
    writeData(data, cnt);
}


void Serializer::writeUInt16s(const uint16_t *data, size_t cnt)
{
    SYRINX_EXPECT(data);
    writeData(data, cnt);
}


void Serializer::writeUInt32s(const uint32_t *data, size_t cnt)
{
    SYRINX_EXPECT(data);
    writeData(data, cnt);
}


void Serializer::writeUInt64s(const uint64_t *data, size_t cnt)
{
    SYRINX_EXPECT(data);
    writeData(data, cnt);
}


void Serializer::writeFloats(const float *data, size_t cnt)
{
    SYRINX_EXPECT(data);
    writeData(data, cnt);
}


void Serializer::writeFloats(const double *data, size_t cnt)
{
    SYRINX_EXPECT(data);
    writeData(data, cnt);
}


void Serializer::writeString(const std::string &str)
{
    writeUInt32(static_cast<uint32_t>(str.size()));
    mDataStream->write(str.c_str(), str.length());
}

} // namespace Syrinx