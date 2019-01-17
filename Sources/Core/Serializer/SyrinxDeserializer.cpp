#include "Serializer/SyrinxDeserializer.h"
#include "Common/SyrinxAssert.h"
#include "Common/SyrinxMemory.h"
#include "Exception/SyrinxException.h"
#include "Logging/SyrinxLogManager.h"

namespace Syrinx {

Deserializer::Deserializer()
    : mDataStream(nullptr)
    , mVersion(SerializerVersionInfo)
    , mEndian(Endian::DEFAULT)
{
    SYRINX_ENSURE(!mDataStream && mEndian == Endian::DEFAULT);
}


void Deserializer::deserializeFileHeader()
{
    if (readUInt32() != SERIALIZER_HEADER_CHECKER) {
		SYRINX_ERROR_FMT("fail to check file header, invalid file header for file [{}]", mDataStream->getName());
		SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::DeserializationError, "invalid file header for file [{}]", mDataStream->getName());
    }

    if (readString() != mVersion) {
        SYRINX_ERROR_FMT("fail to check version info, invalid file header for file {}", mDataStream->getName());
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::DeserializationError, "invalid version info for file [{}]", mDataStream->getName());
    }

    auto fileEndian = static_cast<Endian>(readUInt8());
    if (fileEndian != mEndian) {
        SYRINX_ERROR_FMT("invalid file endian for file [{}]", mDataStream->getName());
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::DeserializationError, "invalid file endian for file [{}]", mDataStream->getName());
    }

    readCustomHeader();
}


void Deserializer::readData(void *data, size_t elemSize, size_t cnt)
{
    SYRINX_EXPECT(mDataStream && data);
    mDataStream->read(data, elemSize * cnt);
}


template <typename T>
void Deserializer::readData(T *data, size_t cnt)
{
    SYRINX_EXPECT(mDataStream && mDataStream->isReadable() && data);
    mDataStream->read(data, sizeof(T) * cnt);
}


void Deserializer::readUInt8s(uint8_t *data, size_t cnt)
{
    SYRINX_EXPECT(data);
    readData(data, cnt);
}


void Deserializer::readUInt16s(uint16_t *data, size_t cnt)
{
    SYRINX_EXPECT(data);
    readData(data, cnt);
}


void Deserializer::readUInt32s(uint32_t *data, size_t cnt)
{
    SYRINX_EXPECT(data);
    readData(data, cnt);
}


void Deserializer::readUInt64s(uint64_t *data, size_t cnt)
{
    SYRINX_EXPECT(data);
    readData(data, cnt);
}


void Deserializer::readFloats(float *data, size_t cnt)
{
    SYRINX_EXPECT(data);
    readData(data, cnt);
}


void Deserializer::readFloats(double *data, size_t cnt)
{
    SYRINX_EXPECT(data);
    readData(data, cnt);
}


bool Deserializer::readBool()
{
    uint8_t value;
    readData(&value, 1);
    SYRINX_EXPECT(value == 0 || value == 1);
    return static_cast<bool>(value);
}


uint8_t Deserializer::readUInt8()
{
    uint8_t value;
    readData(&value, 1);
    return value;
}


uint16_t Deserializer::readUInt16()
{
    uint16_t value;
    readData(&value, 1);
    return value;
}


uint32_t Deserializer::readUInt32()
{
    uint32_t value;
    readData(&value, 1);
    return value;
}


uint64_t Deserializer::readUInt64()
{
    uint64_t value;
    readData(&value, 1);
    return value;
}


std::string Deserializer::readString()
{
    uint32_t length = readUInt32();
    auto *data = SYRINX_NEW_ARRAY(int8_t, length + 1);
    readData(data, length);
    data[length] = '\0';
    std::string ret(reinterpret_cast<char*>(data));
    SYRINX_DELETE_ARRAY(data);
    return ret;
}

} // namespace Syrinx