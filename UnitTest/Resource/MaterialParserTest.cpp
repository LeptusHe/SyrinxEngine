#include <gmock/gmock.h>
#include <ResourceLoader/SyrinxMaterialParser.h>
#include <TestDouble/DefaultDataStream.h>
#include <TestDouble/DefaultFileManager.h>
#include <TestDouble/DefaultProgramStage.h>
#include <TestDouble/DefaultHardwareResourceManager.h>

using namespace testing;
using namespace Syrinx;

/*
class ParseShaderParameterValue : public Test {
public:
    void SetUp() override
    {
        mShaderParameter.setName("parse shader parameter value");
        mFileManager = new DefaultFileManager();
        mHardwareResourceManager = new HardwareResourceManager(mFileManager);
        mParser = new MaterialParser(mFileManager, mHardwareResourceManager);
    }

    void TearDown() override
    {
        delete mFileManager;
        delete mParser;
    }


protected:
    FileManager *mFileManager;
    HardwareResourceManager *mHardwareResourceManager;
    MaterialParser *mParser;
    ShaderParameter mShaderParameter;
};


TEST_F(ParseShaderParameterValue, int_type_parameter)
{
    mShaderParameter.setType("int");
    auto value = mParser->parseShaderParameterValue(mShaderParameter, "123");
    ASSERT_THAT(std::get<int>(value), Eq(123));
}


TEST_F(ParseShaderParameterValue, float_type_parameter)
{
    mShaderParameter.setType("float");
    auto value = mParser->parseShaderParameterValue(mShaderParameter, "1.23");
    ASSERT_THAT(std::get<float>(value), FloatEq(1.23));
}


TEST_F(ParseShaderParameterValue, color_type_parameter)
{
    mShaderParameter.setType("color");
    auto value = mParser->parseShaderParameterValue(mShaderParameter, "0.0 1.0 2.0 3.0");
    auto colorData = static_cast<const float*>(std::get<Color>(value));
    for (int i = 0; i < 4; ++ i) {
        ASSERT_THAT(colorData[i], FloatEq(static_cast<float>(i)));
    }
}




class ParseVertexAttribute : public Test {
public:
    void SetUp() override
    {
        mFileManager = new DefaultFileManager();
        mHardwareResourceManager = new HardwareResourceManager(mFileManager);
        mParser = new MaterialParser(mFileManager, mHardwareResourceManager);
    }

    void TearDown() override
    {
        delete mFileManager;
        delete mParser;
    }

    void setXMLContent(const std::string& content)
    {
        mDocument.load_string(content.c_str());
    }

    std::vector<VertexAttribute> parseVertexAttributeSet()
    {
        auto vertexAttributeSetNode = mDocument.child("input-vertex-attribute-set");
        return mParser->parseVertexAttributeSet(vertexAttributeSetNode);
    }

protected:
    FileManager *mFileManager = nullptr;
    HardwareResourceManager *mHardwareResourceManager = nullptr;
    MaterialParser *mParser = nullptr;
    pugi::xml_document mDocument;
};


TEST_F(ParseVertexAttribute, parse_position_attribute)
{
    std::string xmlContent = "<input-vertex-attribute-set>\n"
                             "    <attribute name=\"inPos\" semantic=\"position\" data-type=\"float3\"/>\n"
                             "</input-vertex-attribute-set>\n";
    setXMLContent(xmlContent);
    auto vertexAttributeSet = parseVertexAttributeSet();

    ASSERT_THAT(vertexAttributeSet.size(), Eq(1));
    const auto& positionAttribute = vertexAttributeSet[0];
    ASSERT_THAT(positionAttribute.getName(), Eq("inPos"));
    ASSERT_THAT(positionAttribute.getSemantic()._value, Eq(VertexAttributeSemantic::Position));
    ASSERT_THAT(positionAttribute.getDataType()._value, Eq(VertexAttributeDataType::FLOAT3));
}


TEST_F(ParseVertexAttribute, parse_normal_attribute)
{
    std::string xmlContent = "<input-vertex-attribute-set>\n"
                             "    <attribute name=\"inNormal\" semantic=\"normal\" data-type=\"float3\"/>\n"
                             "</input-vertex-attribute-set>\n";
    setXMLContent(xmlContent);
    auto vertexAttributeSet = parseVertexAttributeSet();

    ASSERT_THAT(vertexAttributeSet.size(), Eq(1));
    const auto& positionAttribute = vertexAttributeSet[0];
    ASSERT_THAT(positionAttribute.getName(), Eq("inNormal"));
    ASSERT_THAT(positionAttribute.getSemantic()._value, Eq(VertexAttributeSemantic::Normal));
    ASSERT_THAT(positionAttribute.getDataType()._value, Eq(VertexAttributeDataType::FLOAT3));
}


TEST_F(ParseVertexAttribute, parse_tex_coord_attribute)
{
    std::string xmlContent = "<input-vertex-attribute-set>\n"
                             "    <attribute name=\"inTexCoord\" semantic=\"tex-coord\" data-type=\"float2\"/>\n"
                             "</input-vertex-attribute-set>\n";
    setXMLContent(xmlContent);
    auto vertexAttributeSet = parseVertexAttributeSet();

    ASSERT_THAT(vertexAttributeSet.size(), Eq(1));
    const auto& positionAttribute = vertexAttributeSet[0];
    ASSERT_THAT(positionAttribute.getName(), Eq("inTexCoord"));
    ASSERT_THAT(positionAttribute.getSemantic()._value, Eq(VertexAttributeSemantic::TexCoord));
    ASSERT_THAT(positionAttribute.getDataType()._value, Eq(VertexAttributeDataType::FLOAT2));
}




class ParseShaderPass : public Test {
public:
    void SetUp() override
    {
        mFileManager = new DefaultFileManager();
        mHardwareResourceManager = new DefaultHardwareResourceManager(mFileManager);
        mParser = new MaterialParser(mFileManager, mHardwareResourceManager);
    }

    void TearDown() override
    {
        delete mFileManager;
        delete mHardwareResourceManager;
        delete mParser;
    }

    void parseXMLContent(const std::string& content)
    {
        mXMLDocument.load_string(content.c_str());
        mShaderPass = mParser->parseShaderPass(mXMLDocument.child("pass"));
    }

protected:
    FileManager *mFileManager = nullptr;
    HardwareResourceManager *mHardwareResourceManager = nullptr;
    MaterialParser *mParser = nullptr;
    pugi::xml_document mXMLDocument;
    std::unique_ptr<ShaderPass> mShaderPass;
};
*/

/*
TEST_F(ParseShaderPass, valid_pass_name_and_program_name)
{
    const std::string xmlContent = "<pass name=\"shadow\">\n"
                                   "    <vertex-program>\n"
                                   "        <input-vertex-attribute-set>\n"
                                   "            <attribute name=\"inPos\" semantic=\"position\" data-type=\"float3\"/>\n"
                                   "        </input-vertex-attribute-set>\n"
                                   "        <code-file>../shadow_vertex.glsl</code-file>\n"
                                   "    </vertex-program>\n"
                                   "    <fragment-program>\n"
                                   "        <code-file>../shadow_fragment.glsl</code-file>\n"
                                   "    </fragment-program>\n"
                                   "</pass>";

    parseXMLContent(xmlContent);

    ASSERT_THAT(mShaderPass->getName(), Eq("shadow"));
    ASSERT_THAT(mShaderPass->getParameterMap().size(), Eq(0));

    auto vertexProgram = mShaderPass->getProgramStage(ProgramStageType::VertexStage);
    ASSERT_THAT(vertexProgram, NotNull());
    ASSERT_THAT(vertexProgram->getName(), Eq("../shadow_vertex.glsl"));

    auto fragmentProgram = mShaderPass->getProgramStage(ProgramStageType::FragmentStage);
    ASSERT_THAT(fragmentProgram, NotNull());
    ASSERT_THAT(fragmentProgram->getName(), Eq("../shadow_fragment.glsl"));
}




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



class ParseShader : public Test {
public:
    void SetUp() override
    {
        mFileManager = new DefaultFileManager();
        mHardwardResourceManager = new DefaultHardwareResourceManager(mFileManager);
        mParser = new MaterialParser(mFileManager, mHardwardResourceManager);
    }

    void TearDown() override
    {
        delete mFileManager;
        delete mHardwardResourceManager;
        delete mParser;
        delete mShader;
    }

    void parseShaderFile(const std::string& fileName, const std::string& fileContent)
    {
        mFileStream = new FileStreamMock(fileName);
        mFileStream->setContent(fileContent);
        mFileManager->addFileStream(mFileStream);
        mShader = mParser->parseShader(fileName);
    }

protected:
    FileStreamMock *mFileStream = nullptr;
    DefaultFileManager *mFileManager = nullptr;
    HardwareResourceManager *mHardwardResourceManager = nullptr;
    MaterialParser *mParser = nullptr;
    pugi::xml_document mXMLDocument;
    Shader* mShader;
};


TEST_F(ParseShader, valid_shader_parameters)
{
    const std::string xmlContent =
            "<shader name=\"pbr\">\n"
            "    <input-parameter-set>\n"
            "        <parameter name=\"skyColor\" type=\"color\" value=\"1.0 1.0 0.0 1.0\"/>\n"
            "        <parameter name=\"texIndex\" type=\"int\" value=\"1\"/>\n"
            "        <parameter name=\"modelScale\" type=\"float\" value=\"10.0\"/>\n"
            "    </input-parameter-set>\n"
            "\n"
            "    <pass name=\"shadow\">\n"
            "        <vertex-program>\n"
            "            <input-vertex-attribute-set>\n"
            "                <attribute name=\"inPosShadow\" semantic=\"position\" data-type=\"float3\"/>\n"
            "            </input-vertex-attribute-set>\n"
            "            <code-file>../shadow_vertex.glsl</code-file>\n"
            "        </vertex-program>\n"
            "        <fragment-program>\n"
            "            <code-file>../shadow_fragment.glsl</code-file>\n"
            "        </fragment-program>\n"
            "    </pass>\n"
            "\n"
            "    <pass name=\"lighting\">\n"
            "        <vertex-program>\n"
            "            <input-vertex-attribute-set>\n"
            "                <attribute name=\"inPos\" semantic=\"position\" data-type=\"float3\"/>\n"
            "                <attribute name=\"inNormal\" semantic=\"normal\" data-type=\"float3\"/>\n"
            "                <attribute name=\"inTexCoord\" semantic=\"texCoord\" data-type=\"float2\"/>\n"
            "            </input-vertex-attribute-set>\n"
            "            <input-parameter-set>\n"
            "                <parameter ref=\"modelScale\"/>\n"
            "            </input-parameter-set>\n"
            "            <code-file>../lighting_vertex.glsl</code-file>\n"
            "        </vertex-program>\n"
            "        <fragment-program>\n"
            "            <input-parameter-set>\n"
            "                <parameter ref=\"skyColor\"/>\n"
            "                <parameter ref=\"modelScale\"/>\n"
            "            </input-parameter-set>\n"
            "            <code-file>../lighting_fragment.glsl</code-file>\n"
            "        </fragment-program>\n"
            "    </pass>\n"
            "</shader>";

    parseShaderFile("pbr-shader", xmlContent);

    ASSERT_THAT(mShader->getName(), Eq("pbr"));

    auto shaderParameterList = mShader->getShaderParameterList();
    ASSERT_THAT(shaderParameterList.size(), Eq(3));

    auto skyColorParameter = mShader->getShaderParameter("skyColor");
    ASSERT_THAT(skyColorParameter->getName(), Eq("skyColor"));
    ASSERT_THAT(skyColorParameter->getType()._value, Eq(ShaderParameterType::COLOR));

    auto texIndexParameter = mShader->getShaderParameter("texIndex");
    ASSERT_THAT(texIndexParameter->getName(), Eq("texIndex"));
    ASSERT_THAT(texIndexParameter->getType()._value, Eq(ShaderParameterType::INT));
    ASSERT_THAT(std::get<int>(texIndexParameter->getValue()), Eq(1));

    auto modelScale = mShader->getShaderParameter("modelScale");
    ASSERT_THAT(modelScale->getName(), Eq("modelScale"));
    ASSERT_THAT(modelScale->getType()._value, Eq(ShaderParameterType::FLOAT));
    ASSERT_THAT(std::get<float>(modelScale->getValue()), FloatEq(10.0));


    auto shadowPass = mShader->getShaderPass("shadow");
    ASSERT_THAT(shadowPass, NotNull());
    ASSERT_THAT(shadowPass->getName(), Eq("shadow"));
    const auto& vertexAttributeListForShadowPass = shadowPass->getVertexAttributeList();
    ASSERT_THAT(vertexAttributeListForShadowPass.size(), Eq(1));

    auto positionAttributeForShadowPass = shadowPass->getVertexAttribute("inPosShadow");
    ASSERT_THAT(positionAttributeForShadowPass, NotNull());
    ASSERT_THAT(positionAttributeForShadowPass->getName(), Eq("inPosShadow"));

    auto shadowVertexProgram = shadowPass->getProgramStage(ProgramStageType::VertexStage);
    ASSERT_THAT(shadowVertexProgram, NotNull());
    ASSERT_THAT(shadowVertexProgram->getName(), Eq("../shadow_vertex.glsl"));

    auto shadowFragmentProgram = shadowPass->getProgramStage(ProgramStageType::FragmentStage);
    ASSERT_THAT(shadowFragmentProgram, NotNull());
    ASSERT_THAT(shadowFragmentProgram->getName(), Eq("../shadow_fragment.glsl"));


    auto lightingPass = mShader->getShaderPass("lighting");
    ASSERT_THAT(lightingPass, NotNull());
    ASSERT_THAT(lightingPass->getName(), Eq("lighting"));
    const auto& vertexAttributeListForLightingPass = lightingPass->getVertexAttributeList();
    ASSERT_THAT(vertexAttributeListForLightingPass.size(), Eq(3));

    ASSERT_THAT(lightingPass->getVertexAttribute("inPos"), NotNull());
    ASSERT_THAT(lightingPass->getVertexAttribute("inNormal"), NotNull());
    ASSERT_THAT(lightingPass->getVertexAttribute("inTexCoord"), NotNull());

    auto lightingVertexProgram = lightingPass->getProgramStage(ProgramStageType::VertexStage);
    ASSERT_THAT(lightingVertexProgram, NotNull());
    ASSERT_THAT(lightingVertexProgram->getName(), Eq("../lighting_vertex.glsl"));


    auto lightingFragmentProgram = lightingPass->getProgramStage(ProgramStageType::FragmentStage);
    ASSERT_THAT(lightingFragmentProgram, NotNull());
    ASSERT_THAT(lightingFragmentProgram->getName(), Eq("../lighting_fragment.glsl"));

    auto modelScaleRefList = lightingPass->getProgramStageListForReferencedParameter("modelScale");
    ASSERT_THAT(modelScaleRefList, NotNull());
    ASSERT_THAT(modelScaleRefList->size(), Eq(2));

    auto skyColorRefList = lightingPass->getProgramStageListForReferencedParameter("skyColor");
    ASSERT_THAT(skyColorRefList, NotNull());
    ASSERT_THAT(skyColorRefList->size(), Eq(1));

    auto texIndexRefList = lightingPass->getProgramStageListForReferencedParameter("texIndex");
    ASSERT_THAT(texIndexRefList, IsNull());
}
 */