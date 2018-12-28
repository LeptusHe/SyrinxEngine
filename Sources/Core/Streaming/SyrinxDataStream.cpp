#include "Logging/SyrinxLogManager.h"
#include "Streaming/SyrinxDataStream.h"

namespace Syrinx {

DataStream::DataStream(const std::string& name) : mName(name), mSize(0)
{
    SYRINX_ENSURE(!mName.empty());
    SYRINX_ENSURE(mSize == 0);
}


DataStream::DataStream(const std::string& name, size_t size) : mName(name), mSize(size)
{
    SYRINX_ENSURE(!mName.empty());
    SYRINX_ENSURE(mSize > 0);
}


size_t DataStream::read(void *buffer, size_t byteSize)
{
    SYRINX_WARN("unimplemented method called");
    return 0;
}


size_t DataStream::write(const void *buffer, size_t byteSize)
{
    SYRINX_WARN("unimplemented method called");
    return 0;
}


std::string DataStream::getName() const
{
    return mName;
}


void DataStream::setSize(size_t size)
{
    mSize = size;
}


size_t DataStream::getSize() const
{
    return mSize;
}

} // namespace Syrinx