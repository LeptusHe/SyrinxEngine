#include <gmock/gmock.h>
#include <RenderResource/SyrinxModel.h>
#include <ResourceManager/SyrinxModelManager.h>
#include <TestDouble/DefaultFileManager.h>
#include <TestDouble/DefaultDataStream.h>

using namespace testing;
using namespace Syrinx;

/*
namespace {

class FileStreamMock : public DefaultDataStream {
public:
    explicit FileStreamMock(const std::string& name) : DefaultDataStream(name) { }
    void setContent(const std::string& content) { mContent = content; }
    std::string getAsString() override { return mContent; }

private:
    std::string mContent;
};

} // anonymous namespace


TEST(Model, parse_json_file_to_get_mesh_set)
{
    const std::string modelFileName = "cube.smodel";
    const std::string jsonContent = "["
                                    "     {\"name\":\"mesh0\", \"path\":\"./mesh0.smodel\"},"
                                    "     {\"name\":\"mesh1\", \"path\":\"./mesh1.smodel\"}"
                                    "]";

    auto fileManager = std::make_unique<DefaultFileManager>();
    auto modelFileStream = std::make_unique<FileStreamMock>(modelFileName);
    modelFileStream->setContent(jsonContent);

    auto mesh0FileStream = std::make_unique<FileStreamMock>("/./mesh0.smodel");
    auto mesh1FileStream = std::make_unique<FileStreamMock>("/./mesh1.smodel");

    fileManager->addFileStream(modelFileStream.get());
    fileManager->addFileStream(mesh0FileStream.get());
    fileManager->addFileStream(mesh1FileStream.get());


    auto modelManager = std::make_unique<ModelManager>(fileManager.get(), )
    Model model(modelFileName, fileManager, new HardwareResourceManager(fileManager));

    ASSERT_THAT(model.getName(), Eq("cube.smodel"));

    const auto& meshMap = model.getMeshMap();
    ASSERT_THAT(meshMap.size(), Eq(0));
}
*/