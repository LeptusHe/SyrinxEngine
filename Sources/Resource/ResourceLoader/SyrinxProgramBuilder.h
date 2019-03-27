#pragma once
#include <deque>
#include "ResourceLoader/SyrinxProgramParser.h"

namespace Syrinx {

class ProgramBuilder {
public:
    explicit ProgramBuilder(FileManager *fileManager);
    ~ProgramBuilder() = default;

    ProgramBuilder& addPredefinedMacro(const std::string& macroName);
    ProgramBuilder& addPredefinedMacro(const std::string& macroName, const std::string& value);
    std::string build(const std::string& fileName);

private:
    void clear();

private:
    FileManager *mFileManager;
    std::vector<std::string> mMacroStringList;
};

} // namespace Syrinx
