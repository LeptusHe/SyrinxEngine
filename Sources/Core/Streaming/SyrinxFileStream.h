#pragma once
#include "Streaming/SyrinxDataStream.h"

namespace Syrinx {

enum class FileAccessMode {
    READ, WRITE
};


class FileStream : public DataStream {
public:
    FileStream(const std::string& fileName, FileAccessMode accessMode) noexcept(false);
    FileStream(const FileStream&) = delete;
    FileStream& operator=(const FileStream&) = delete;
    ~FileStream() override;

    size_t read(void *buffer, size_t byteSize) override;
    size_t write(const void *buffer, size_t byteSize) override;
    std::string getLine() override;
    bool getLine(std::string& line) override;
    std::string getAsString() override;
    void skip(uint64_t byteSize) override;
    void seek(size_t pos) override;
    size_t tell() const override;
    bool eof() const override;
    bool isReadable() const override;
    bool isWriteable() const override;

private:
    void close();

private:
    std::fstream *mFileStream;
    FileAccessMode mAccessMode;
};

} // namespace Syrinx