#include <gmock/gmock.h>
#include <ResourceLoader/SyrinxProgramParser.h>
#include <TestDouble/DefaultDataStream.h>

using namespace testing;
using namespace Syrinx;

namespace {

class FileStreamMock : public DefaultDataStream {
public:
    explicit FileStreamMock(const std::string& name) : DefaultDataStream(name), mLines() { }
    void setLines(const std::vector<std::string>& lines) { mLines = lines; }
    bool getLine(std::string& line) override
    {
        if (mLineCounter < mLines.size()) {
            line = mLines[mLineCounter];
            mLineCounter += 1;
            return true;
        }
        return false;
    }

private:
    std::vector<std::string> mLines;
    int mLineCounter = 0;
};


class FileManagerMock : public FileManager {
public:
    ~FileManagerMock() override
    {
        for (const auto& [name, fileStream] : mFileStreamMap) {
            delete fileStream;
        }
    }

    DataStream *openFile(const std::string& fileName, FileAccessMode accessMode) const override
    {
        auto iter = mFileStreamMap.find(fileName);
        if (iter != std::end(mFileStreamMap)) {
            return iter->second;
        }
        return nullptr;
    }

    void addFileStream(FileStreamMock *fileStream)
    {
        SYRINX_EXPECT(fileStream);
        mFileStreamMap[fileStream->getName()] = fileStream;
    }

private:
    std::unordered_map<std::string, FileStreamMock *> mFileStreamMap;
};

} // anonymous namespace



TEST(ProgramParser, parse_program_that_does_not_include_file)
{
    const std::string programFileName = "main.frag";
    std::vector<std::string> contentsOfMainProgram = {" layout(location = 0) in vec3 inPos;"};

    auto programFileStream = new FileStreamMock(programFileName);
    programFileStream->setLines(contentsOfMainProgram);

    auto fileManager = new FileManagerMock();
    fileManager->addFileStream(programFileStream);

    ProgramParser parser(programFileName, fileManager);
    ASSERT_THAT(parser.getSource(), Eq(contentsOfMainProgram[0] + '\n'));
    ASSERT_TRUE(parser.getIncludedFileList().empty());
}


TEST(ProgramParserTest, parse_program_that_include_file)
{
    const std::string programFileName = "main.frag";
    const std::vector<std::string> contentsOfMainProgram = {
            " #include <math.glsl> \n",
            " layout(location = 0) in vec3 inPos;\n"
    };
    auto programFileStream = new FileStreamMock(programFileName);
    programFileStream->setLines(contentsOfMainProgram);


    const std::string includedFileName = "math.glsl";
    auto includedFileStream = new FileStreamMock(includedFileName);
    const std::vector<std::string> contentsOfIncludedFile = {
            "layout(location =0) in vec4 inMath;\n"
    };
    includedFileStream->setLines(contentsOfIncludedFile);


    auto fileManager = new FileManagerMock();
    fileManager->addFileStream(programFileStream);
    fileManager->addFileStream(includedFileStream);

    ProgramParser parser(programFileName, fileManager);

    auto source = contentsOfIncludedFile[0] + '\n' + contentsOfMainProgram[1] + '\n';
    ASSERT_THAT(parser.getSource(), Eq(source));
    ASSERT_THAT(parser.getIncludedFileList().size(), Eq(1));
    ASSERT_THAT(parser.getIncludedFileList()[0], Eq(includedFileName));
}
