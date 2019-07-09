#include <gmock/gmock.h>
#include <memory>
#include <Exception/SyrinxException.h>
#include <Scene/SyrinxSceneImporter.h>
#include <ResourceManager/SyrinxModelManager.h>
#include <ResourceManager/SyrinxMaterialManager.h>
#include <TestDouble/DefaultDataStream.h>
#include <TestDouble/DefaultFileManager.h>
#include <TestDouble/DefaultHardwareResourceManager.h>

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


class ModelManagerMock : public ModelManager {
public:
    explicit ModelManagerMock(FileManager *fileManager, HardwareResourceManager *hardwareResourceManager)
        : ModelManager(fileManager, hardwareResourceManager)
    {

    }

    ~ModelManagerMock() override = default;

    Model* createModel(const std::string& fileName) override
    {
        auto model = new Model(fileName, getFileManager(), new HardwareResourceManager(getFileManager()));
        mModel = std::unique_ptr<Model>(model);
        return model;
    }

    MOCK_METHOD1(findModel, Model*(const std::string&));

private:
    std::unique_ptr<Model> mModel;
};


class MaterialManagerMock : public MaterialManager {
public:
    explicit MaterialManagerMock(FileManager *fileManager, HardwareResourceManager *hardwareResourceManager)
        : MaterialManager(fileManager, hardwareResourceManager) { }
    ~MaterialManagerMock() override = default;

    Material* createMaterial(const std::string& fileName) override
    {
        mMaterial = std::make_unique<Material>(fileName);
        return mMaterial.get();
    }

private:
    std::unique_ptr<Material> mMaterial;
};

} // anonymous namespace


class LoadingScene : public Test {
public:
    void SetUp() override
    {
        mFileManager = new DefaultFileManager();
        mHardwareResourceManager = new HardwareResourceManager(mFileManager);
        mModelManager = new ModelManagerMock(mFileManager, mHardwareResourceManager);
        mMaterialManager = new MaterialManagerMock(mFileManager, mHardwareResourceManager);
        mSceneManager = new SceneManager(mModelManager, mMaterialManager);
        mSceneLoader = new SceneLoader(mSceneManager);
    }

    void TearDown() override
    {
        delete mFileManager;
        delete mModelManager;
        delete mMaterialManager;
        delete mSceneLoader;
        delete mSceneManager;
    }

    SceneNode* loadScene(const std::string& fileContent)
    {
        mFileStream.setContent(fileContent);
        return mSceneLoader->loadScene(mFileStream);
    }

private:
    FileStreamMock mFileStream{"simple.scene"};
    FileManager *mFileManager = nullptr;
    HardwareResourceManager *mHardwareResourceManager = nullptr;
    ModelManager *mModelManager = nullptr;
    MaterialManager *mMaterialManager = nullptr;
    SceneManager *mSceneManager = nullptr;
    SceneLoader *mSceneLoader = nullptr;
};


TEST_F(LoadingScene, invalid_scene_file_format)
{
    ASSERT_THROW(loadScene(""), InvalidParamsException);
}


TEST_F(LoadingScene, load_empty_scene)
{
    std::string fileContent = "<scene></scene>";
    auto rootNode = loadScene(fileContent);

    ASSERT_THAT(rootNode, NotNull());
    ASSERT_THAT(rootNode->getName(), Eq("root"));
    ASSERT_THAT(rootNode->getParent(), IsNull());
    ASSERT_THAT(rootNode->getNumChild(), Eq(0));
}


TEST_F(LoadingScene, load_entity)
{
    std::string sceneFileContent = "<scene>\n"
                                   "    <node name=\"robot-model\">\n"
                                   "        <position x=\"1.0\" y=\"2.0\" z=\"0.0\"/>\n"
                                   "        <scale x=\"1\" y=\"1\" z=\"1\"/>\n"
                                   "        <entity name=\"robot\" model-file=\"robot.smodel\" material-file=\"pbr.smat\"/>\n"
                                   "    </node>\n"
                                   "</scene>";
    auto rootNode = loadScene(sceneFileContent);
    ASSERT_THAT(rootNode->getNumChild(), Eq(1));

    auto robotNode = rootNode->getChild("robot-model");
    ASSERT_THAT(robotNode, NotNull());
    ASSERT_THAT(robotNode->getName(), Eq("robot-model"));

    const auto& robotPosition = robotNode->getPosition();
    ASSERT_THAT(robotPosition.x, FloatEq(1.0));
    ASSERT_THAT(robotPosition.y, FloatEq(2.0));
    ASSERT_THAT(robotPosition.z, FloatEq(0.0));

    auto entity = robotNode->getEntity();
    ASSERT_THAT(entity, NotNull());
    ASSERT_THAT(entity->getName(), Eq("robot"));
}
*/