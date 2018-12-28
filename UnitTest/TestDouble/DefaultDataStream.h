#pragma once
#include <Streaming/SyrinxDataStream.h>


class DefaultDataStream : public Syrinx::DataStream {
public:
    explicit DefaultDataStream(const std::string& name) : DataStream(name) {}
    DefaultDataStream(const std::string& name, size_t size) : DataStream(name, size) {}
    ~DefaultDataStream() override = default;

    size_t read(void *buffer, size_t byteSize) override { return 0; }
    size_t write(const void *buffer, size_t byteSize) override { return 0; }
    std::string getLine() override { return {}; }
    bool getLine(std::string& line) override { return false; }
    std::string getAsString() override { return {}; }
    void skip(uint64_t byteSize) override { }
    size_t tell() const override { return 0; }
    void seek(size_t pos) override {  }
    bool eof() const override { return true; }
    bool isReadable() const override { return true; }
    bool isWriteable() const override { return true; }
};