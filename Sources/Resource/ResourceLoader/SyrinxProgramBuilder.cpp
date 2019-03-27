#include "ResourceLoader/SyrinxProgramBuilder.h"

namespace Syrinx {

ProgramBuilder::ProgramBuilder(FileManager *fileManager)
    : mFileManager(fileManager)
    , mMacroStringList()
{
    SYRINX_ENSURE(mFileManager);
    SYRINX_ENSURE(mMacroStringList.empty());
}


ProgramBuilder& ProgramBuilder::addPredefinedMacro(const std::string& macroName)
{
    SYRINX_EXPECT(!macroName.empty());
    mMacroStringList.push_back("#define " + macroName + "\n");
    return *this;
}


ProgramBuilder& ProgramBuilder::addPredefinedMacro(const std::string& macroName, const std::string& value)
{
    SYRINX_EXPECT(!macroName.empty());
    SYRINX_EXPECT(!value.empty());
    mMacroStringList.push_back("#define " + macroName + " " + value + "\n");
    return *this;
}


std::string ProgramBuilder::build(const std::string& fileName)
{
    SYRINX_EXPECT(!fileName.empty());
    ProgramParser programParser(fileName, mFileManager);
    const auto& source = programParser.getSource();

    std::string result = "#version 450 core\n";
    for (const auto& macroString : mMacroStringList) {
        result += macroString;
    }
    clear();
    SYRINX_ENSURE(mMacroStringList.empty());
    return result + source;
}


void ProgramBuilder::clear()
{
    mMacroStringList.clear();
}

} // namespace Syrinx
