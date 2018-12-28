#include "Exception/SyrinxException.h"
#include <cassert>
#include <fmt/format.h>

namespace Syrinx {

Exception::Exception(const std::string& type, const std::string& description, const std::string& file, const std::string& source, long line)
    : exception()
    , mType(type)
    , mDescription(description)
    , mMessage()
    , mFile(file)
    , mSource(source)
    , mLine(line)
{
    assert(!description.empty() && !type.empty());
    assert(!file.empty() && !source.empty() && line >= 0);
    mMessage = buildMessage();
    assert(!mMessage.empty());
}


const char* Exception::what() const noexcept
{
    return mMessage.c_str();
}


std::string Exception::getType() const
{
    return mType;
}


std::string Exception::getFile() const
{
    return mFile;
}


std::string Exception::getSource() const
{
    return mSource;
}


long Exception::getLine() const
{
    return mLine;
}


std::string Exception::getDescription() const
{
    return mDescription;
}


void ExceptionFactory::throwException(Syrinx::ExceptionCode code,
                                      const std::string& description,
                                      const std::string& file,
                                      const std::string& source,
                                      long line)
{
    switch (code) {
        case ExceptionCode::FileNotFound: throw FileNotFoundException(description, file, source, line);
        case ExceptionCode::FileSystemError: throw FileSystemException(description, file, source, line);
        case ExceptionCode::ImageLoadError: throw ImageLoadException(description, file, source, line);
        case ExceptionCode::SerializationError:
        case ExceptionCode::DeserializationError: throw SerializationException(description, file, source, line);
        case ExceptionCode::RuntimeAssertFailure: throw RuntimeAssertionException(description, file, source, line);
        case ExceptionCode::InvalidState: throw InvalidStateException(description, file, source, line);
        case ExceptionCode::InvalidParams: throw InvalidParamsException(description, file, source, line);
        default: assert(false && "undefined exception code");
    }
}


std::string Exception::buildMessage() const
{
    return fmt::format("exception type=[{}], description=[{}], file=[{}], function=[{}], line=[{}]", mType, mDescription, mFile, mSource, mLine);
}

} // namespace Syrinx
