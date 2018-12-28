#include <GL/glew.h>
#include <better-enums/enum.h>
#include <ResourceManager/SyrinxFileManager.h>
#include <HardwareResource/SyrinxProgramPipeline.h>
#include <RenderResource/SyrinxRenderTexture.h>

namespace Syrinx { namespace Tool{

BETTER_ENUM(MapType, uint8_t,
            IrradianceMap,
            PreFilteredMap,
			BrdfIntegrationMap
);


class GenPreComputedIBLImages {
public:
	GenPreComputedIBLImages(GLuint environmentMap, int environmentMapWidth);
	~GenPreComputedIBLImages();


private:
    const std::string loadProgramSource(const std::string& fileName);
    void createProgramPipeline(MapType mapType, const std::string& vertexShaderFileName, const std::string& fragmentShaderFileName);
    void initProgramPipeline();
    void drawCube();
    void drawQuad();
    void genCubeMap(MapType mapType);
    void genIrradianceMap();
    void genPreFilteredMap();
    void genBrdfIntegrationMap();
    void saveFrameData2Image(const std::string& imagePath, int width, int height);

private:
    GLuint mEnvironmentMap;
	int mEnvironmentMapWidth;
	FileManager *mFileManager;

	int mWidth[3];
	int mHeight[3];
    ProgramStage *mVertexProgram[3];
    ProgramStage *mFragmentProgram[3];
    ProgramPipeline *mPipeline[3];

    std::string mSaveFileDirectory;
    std::string mMapTypeName[3];
    std::string mCubeMapFaceName[6];
    std::string mImageType;
};

} // namespace Tool

} // namespace SyrinxGenPreComputedIBLImages