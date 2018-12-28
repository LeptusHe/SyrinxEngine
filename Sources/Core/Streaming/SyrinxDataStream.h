#pragma once
#include <cstdint>
#include <string>

namespace Syrinx {

class DataStream {
public:
    explicit DataStream(const std::string& name);
    DataStream(const std::string& name, size_t size);
    DataStream(const DataStream&) = delete;
    DataStream& operator=(const DataStream&) = delete;
    virtual ~DataStream() = default;

    virtual size_t read(void *buffer, size_t byteSize) = 0;
    virtual size_t write(const void *buffer, size_t byteSize) = 0;
    virtual std::string getLine() = 0;
    virtual bool getLine(std::string& line) = 0;
    virtual std::string getAsString() = 0;
    virtual void skip(uint64_t byteSize) = 0;
    virtual size_t tell() const = 0;
    virtual void seek(size_t pos) = 0;
    virtual bool eof() const = 0;
    virtual bool isReadable() const = 0;
    virtual bool isWriteable() const = 0;
    size_t getSize() const;
    std::string getName() const;

protected:
    void setSize(size_t size);

private:
    std::string mName;
    size_t mSize;
};

} // namespace Syrinx