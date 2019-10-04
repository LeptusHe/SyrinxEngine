#pragma once
#include <exception>
#include <string>
#include <better-enums/enum.h>
#include <fmt/format.h>
#include "Common/SyrinxMacro.h"

namespace Syrinx {

class Exception : public std::exception {
public:
    Exception(const std::string& type, const std::string& description, const std::string& file, const std::string& source, long line);
    ~Exception() override = default;

    const char* what() const noexcept override;
    std::string getType() const;
    std::string getFile() const;
    std::string getSource() const;
    long getLine() const;
    std::string getDescription() const;

protected:
    std::string buildMessage() const;

protected:
    std::string mType;
    std::string mDescription;
    std::string mMessage;
    std::string mFile;
    std::string mSource;
    long mLine;
};


class FileNotFoundException : public Exception {
public:
    FileNotFoundException(const std::string& description, const std::string& file, const std::string& source, long line)
        : Exception(__FUNCTION__, description, file, source, line) { }
};


class FileSystemException : public Exception {
public:
    FileSystemException(const std::string& description, const std::string& file, const std::string& source, long line)
        : Exception(__FUNCTION__, description, file, source, line) { }
};


class ImageLoadException : public Exception {
public:
    ImageLoadException(const std::string& description, const std::string& file, const std::string& source, long line)
        : Exception(__FUNCTION__, description, file, source, line) { }
};


class SerializationException : public Exception {
public:
    SerializationException(const std::string& description, const std::string& file, const std::string& source, long line)
        : Exception(__FUNCTION__, description, file, source, line) { }
};


class RuntimeAssertionException : public Exception {
public:
    RuntimeAssertionException(const std::string& description, const std::string& file, const std::string& source, long line)
        : Exception(__FUNCTION__, description, file, source, line) { }
};


class InvalidStateException : public Exception {
public:
    InvalidStateException(const std::string& description, const std::string& file, const std::string& source, long line)
            : Exception(__FUNCTION__, description, file, source, line) { }
};


class InvalidParamsException : public Exception {
public:
    InvalidParamsException(const std::string& description, const std::string& file, const std::string& source, long line)
            : Exception(__FUNCTION__, description, file, source, line) { }
};


class CUDAException : public Exception {
public:
    CUDAException(const std::string& description, const std::string& file, const std::string& source, long line)
            : Exception(__FUNCTION__, description, file, source, line) {}
};

BETTER_ENUM(ExceptionCode, uint8_t,
    FileNotFound,
    FileSystemError,
    ImageLoadError,
    SerializationError,
    DeserializationError,
    RuntimeAssertFailure,
    InvalidState,
    InvalidParams,
    CUDAError
);


class ExceptionFactory {
public:
    ExceptionFactory() = default;
    ~ExceptionFactory() = default;

    static void throwException(ExceptionCode code, const std::string& description, const std::string& file, const std::string& source, long line);
};

} // namespace Syrinx

#define SYRINX_THROW_EXCEPTION(exceptionCode, description)  Syrinx::ExceptionFactory::throwException(exceptionCode, description, __EXCEPTION_SITE__)
#define SYRINX_THROW_EXCEPTION_FMT(exceptionCode, fmtStr, ...)  Syrinx::ExceptionFactory::throwException(exceptionCode, fmt::format(fmtStr, __VA_ARGS__), __EXCEPTION_SITE__)