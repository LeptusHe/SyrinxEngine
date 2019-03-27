#include "ResourceLoader/SyrinxProgramParser.h"
#include <fstream>
#include <Common/SyrinxAssert.h>
#include <Exception/SyrinxException.h>

namespace Syrinx {

ProgramParser::ProgramParser(const std::string& fileName, FileManager *fileManager)
    : mFileName(fileName)
    , mIncludedFileList()
    , mSource()
    , mFileManager(fileManager)
{
    SYRINX_ENSURE(!mFileName.empty());
    SYRINX_ENSURE(mFileManager);
    mSource = parseFile(mFileName);
    SYRINX_ENSURE(!mSource.empty());
}


const std::string& ProgramParser::getSource() const
{
    return mSource;
}


const std::vector<std::string>& ProgramParser::getIncludedFileList() const
{
    return mIncludedFileList;
}


std::string ProgramParser::parseFile(const std::string& fileName)
{
    auto fileStream = mFileManager->openFile(fileName, FileAccessMode::READ);
    if (!fileStream) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileNotFound, "can not find file [{}]", fileName);
    }

    std::string line;
    std::string source;
    while (fileStream && fileStream->getLine(line)) {
        auto [hasIncludedFile, includedFile] = parseLine(line);
        if (hasIncludedFile) {
            mIncludedFileList.push_back(includedFile);
            source += parseFile(includedFile);
        } else {
            source += line + "\n";
        }
    }
    return source;
}


std::pair<bool, std::string> ProgramParser::parseLine(const std::string& line) const
{
    size_t index = 0;
    size_t size = line.size();

    index = ignoreBlankCharacter(line, index);

    const std::string includeStr = "#include";
    int count = 0;
    while ((index < size) && (count < includeStr.size())){
        if (line[index] != includeStr[count]) {
            return {false, ""};
        }
        count++;
        index++;
    }
    if (count != includeStr.size()) {
        return {false, ""};
    }

    index = ignoreBlankCharacter(line, index);
    if ((index < size) && (line[index++] != '<')) {
        return {false, ""};
    }
    std::string includedFile;
    while ((index < size) && (line[index] != '>')) {
        includedFile.push_back(line[index]);
        index++;
    }
    if (line[index++] != '>') {
        return {false, ""};
    }
    index = ignoreBlankCharacter(line, index);
    if (index == size) {
        return {true, includedFile};
    }
    return {false, ""};
}


size_t ProgramParser::ignoreBlankCharacter(const std::string& str, size_t start) const
{
    SYRINX_EXPECT(start <= str.size());
    auto index = start;
    const auto size = str.size();
    while ((index < size) && (std::isblank(str[index]) || std::isspace(str[index]))) {
        index++;
    }
    return index;
}

} // namespace Syrinx
