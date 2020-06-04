#include "SyrinxOptixResourceManager.h"
#include "SyrinxRadianceAssert.h"

namespace Syrinx {

OptixResourceManager::OptixResourceManager(OptixContext *optixContext, FileManager *fileManager, std::unique_ptr<ImageReader>&& imageReader)
    : mOptixContext(optixContext)
    , mFileManager(fileManager)
    , mImageReader(std::move(imageReader))
    , mBufferCache()
    , mTextureCache()
    , mModuleCache()
    , mProgramGroupCache()
    , mPipelineCache()
{
    initOptions();
    SYRINX_ENSURE(mOptixContext);
    SYRINX_ENSURE(mFileManager);
    SYRINX_ENSURE(mImageReader);
}


CudaBuffer* OptixResourceManager::createBuffer(const std::string& name, size_t size)
{
    OptixProgramGroup tst;
    SYRINX_EXPECT(!name.empty());
    SYRINX_EXPECT(!findBuffer(name));
    auto cudaBuffer = new CudaBuffer(name);
    cudaBuffer->allocate(size);
    mBufferCache.add(name, cudaBuffer);
    SYRINX_ENSURE(findBuffer(name) == cudaBuffer);
    return cudaBuffer;
}


CudaBuffer* OptixResourceManager::createBuffer(const std::string& name, uint8_t *source, size_t size)
{
    SYRINX_EXPECT(source && size > 0);
    auto cudaBuffer = createBuffer(name, size);
    cudaBuffer->upload(source, size);
    return cudaBuffer;
}


CudaBuffer* OptixResourceManager::findBuffer(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    return mBufferCache.find(name);
}


CudaTexture* OptixResourceManager::createTexture2D(std::string& fileFullPath, ImageFormat format, bool enableMipmap)
{
    SYRINX_EXPECT(!fileFullPath.empty());
    SYRINX_EXPECT(!findTexture(fileFullPath));
    Image image = mImageReader->read(fileFullPath, format);
    auto cudaTexture = new CudaTexture(fileFullPath);
    cudaTexture->setType(TextureType::TEXTURE_2D);
    cudaTexture->setPixelFormat(PixelFormat::_from_string(format._to_string()));
    cudaTexture->setWidth(image.getWidth());
    cudaTexture->setHeight(image.getHeight());
    cudaTexture->setDepth(1);
    cudaTexture->enableMipmap(enableMipmap);

    SamplingSetting samplingSetting;
    samplingSetting.setWrapSMethod(TextureWrapMethod::REPEAT);
    samplingSetting.setWrapTMethod(TextureWrapMethod::REPEAT);
    samplingSetting.setWrapRMethod(TextureWrapMethod::REPEAT);
    samplingSetting.setMinFilterMethod(TextureMinFilterMethod::LINEAR_MIPMAP_LINEAR);
    samplingSetting.setMagFilterMethod(TextureMagFilterMethod::LINEAR);

    cudaTexture->setSamplingSetting(samplingSetting);
    cudaTexture->create();

    mTextureCache.add(fileFullPath, cudaTexture);
    SYRINX_ENSURE(findTexture(fileFullPath) == cudaTexture);
    return cudaTexture;
}


CudaTexture* OptixResourceManager::findTexture(std::string& name) const
{
    SYRINX_EXPECT(!name.empty());
    return mTextureCache.find(name);
}


OptixProgramGroup OptixResourceManager::createRayGenProgramGroup(const std::string& name, const OptixResourceManager::ProgramInfo& programInfo)
{
    SYRINX_EXPECT(programInfo.isValid());
    auto module = createOrRetrieveModule(programInfo.fileName);

    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    desc.raygen.module = module;
    desc.raygen.entryFunctionName = programInfo.entryFunctionName.c_str();
    return createProgramGroup(name, desc);
}


OptixProgramGroup OptixResourceManager::createMissProgramGroup(const std::string& name, const OptixResourceManager::ProgramInfo& programInfo)
{
    SYRINX_EXPECT(programInfo.isValid());
    auto module = createOrRetrieveModule(programInfo.fileName);

    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    desc.miss.module = module;
    desc.miss.entryFunctionName = programInfo.entryFunctionName.c_str();
    return createProgramGroup(name, desc);
}


OptixProgramGroup OptixResourceManager::createHitProgramGroup(
    const std::string& name,
    const OptixResourceManager::ProgramInfo& closestHitProgramInfo,
    const OptixResourceManager::ProgramInfo& anyHitProgramInfo,
    const OptixResourceManager::ProgramInfo& intersectProgramInfo)
{
    SYRINX_EXPECT(closestHitProgramInfo.isValid());
    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

    auto closestHitProgramModule = createOrRetrieveModule(closestHitProgramInfo.fileName);
    desc.hitgroup.moduleCH = closestHitProgramModule;
    desc.hitgroup.entryFunctionNameCH = closestHitProgramInfo.entryFunctionName.c_str();

    if (anyHitProgramInfo.isValid()) {
        auto anyHitProgramModule = createOrRetrieveModule(anyHitProgramInfo.fileName);
        desc.hitgroup.moduleAH = anyHitProgramModule;
        desc.hitgroup.entryFunctionNameAH = anyHitProgramInfo.entryFunctionName.c_str();
    }

    if (intersectProgramInfo.isValid()) {
        auto intersectProgramModule = createOrRetrieveModule(intersectProgramInfo.fileName);
        desc.hitgroup.moduleIS = intersectProgramModule;
        desc.hitgroup.entryFunctionNameIS = intersectProgramInfo.entryFunctionName.c_str();
    }
    return createProgramGroup(name, desc);
}


OptixProgramGroup OptixResourceManager::createCallableProgramGroup(
    const std::string& name,
    const OptixResourceManager::ProgramInfo& directCallableProgramInfo,
    const OptixResourceManager::ProgramInfo& continuationCallableProgramInfo)
{
    SYRINX_EXPECT(directCallableProgramInfo.isValid() || continuationCallableProgramInfo.isValid());

    OptixProgramGroupCallables programGroupCallables = {};
    if (directCallableProgramInfo.isValid()) {
        auto module = createOrRetrieveModule(directCallableProgramInfo.fileName);
        programGroupCallables.moduleDC = module;
        programGroupCallables.entryFunctionNameDC = directCallableProgramInfo.entryFunctionName.c_str();
    }

    if (continuationCallableProgramInfo.isValid()) {
        auto module = createOrRetrieveModule(continuationCallableProgramInfo.fileName);
        programGroupCallables.moduleCC = module;
        programGroupCallables.entryFunctionNameCC = continuationCallableProgramInfo.entryFunctionName.c_str();
    }

    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    desc.callables = programGroupCallables;
    return createProgramGroup(name, desc);
}


OptixProgramGroup OptixResourceManager::createExceptionProgramGroup(const std::string& name, const OptixResourceManager::ProgramInfo& exceptionProgramInfo)
{
    SYRINX_EXPECT(exceptionProgramInfo.isValid());
    auto module = createOrRetrieveModule(exceptionProgramInfo.fileName);

    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
    desc.exception.module = module;
    desc.exception.entryFunctionName = exceptionProgramInfo.entryFunctionName.c_str();
    return createProgramGroup(name, desc);
}


OptixProgramGroup OptixResourceManager::findProgramGroup(const std::string& name) const
{
    SYRINX_EXPECT(!name.empty());
    auto iter = mProgramGroupCache.find(name);
    return (iter == std::end(mProgramGroupCache)) ? nullptr : iter->second;
}


OptixPipeline OptixResourceManager::createPipeline(const std::string& name, const std::vector<OptixProgramGroup>& programGroupList)
{
    SYRINX_EXPECT(!name.empty() && !programGroupList.empty());
    SYRINX_EXPECT(!findPipeline(name));

    auto pipeline = mOptixContext->createPipeline(programGroupList, pipelineCompileOptions, pipelineLinkOptions);
    mPipelineCache[name] = pipeline;
    SYRINX_ENSURE(findPipeline(name));
    return pipeline;
}


OptixPipeline OptixResourceManager::findPipeline(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    auto iter = mPipelineCache.find(name);
    return (iter == std::end(mPipelineCache)) ? nullptr : iter->second;
}


void OptixResourceManager::initOptions()
{
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    moduleCompileOptions.maxRegisterCount = 0;

    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG;

    pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    pipelineLinkOptions.maxTraceDepth = 10;
    pipelineLinkOptions.overrideUsesMotionBlur = false;
}


OptixModule OptixResourceManager::createOrRetrieveModule(const std::string& fileName)
{
    auto [fileExist, filePath] = mFileManager->findFile(fileName);
    if (!fileExist) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileNotFound,
            "fail to create OptiX model from file [{}] because it can not be found", fileName);
    }

    auto module = findModule(filePath);
    if (module) {
        return module;
    } else {
        return createModule(filePath);
    }
}


OptixModule OptixResourceManager::createModule(const std::string& filePath)
{
    SYRINX_EXPECT(!filePath.empty());
    SYRINX_EXPECT(!findModule(filePath));
    auto fileStream = mFileManager->openFile(filePath, FileAccessMode::READ);
    if (!fileStream) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileSystemError,
            "fail to create module because the failure to open file [{}]", filePath);
    }

    auto module = mOptixContext->createModule(fileStream->getAsString(), moduleCompileOptions, pipelineCompileOptions);
    SYRINX_ASSERT(module);
    mModuleCache[filePath] = module;
    SYRINX_ENSURE(findModule(filePath));
    return module;
}


OptixModule OptixResourceManager::findModule(const std::string& filePath) const
{
    SYRINX_EXPECT(!filePath.empty());
    auto iter = mModuleCache.find(filePath);
    return (iter == std::end(mModuleCache)) ? nullptr : iter->second;
}


OptixProgramGroup OptixResourceManager::createProgramGroup(const std::string& name, const OptixProgramGroupDesc& desc)
{
    SYRINX_EXPECT(!name.empty() && !findProgramGroup(name));

    auto programGroup = mOptixContext->createProgramGroup(desc);
    SYRINX_ASSERT(programGroup);

    mProgramGroupCache[name] = programGroup;
    SYRINX_ENSURE(findProgramGroup(name));
    return programGroup;
}

AccelerationStructure* OptixResourceManager::createAccelerationStructure(const std::string& name, const std::vector<OptixBuildInput>& buildInputList)
{
    SYRINX_EXPECT(!name.empty() && !buildInputList.empty() && !findAccelerationStructure(name));

    if (findAccelerationStructure(name)) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "create geometry accelerate structure with same name [{}]", name);
    }

    AccelerationStructure accelerateStructure = mOptixContext->buildAccelerationStructure(buildInputList);
    auto result = new AccelerationStructure(std::move(accelerateStructure));
    mAccelerationStructureCache.add(name, result);
    return result;
}

AccelerationStructure* OptixResourceManager::findAccelerationStructure(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    return mAccelerationStructureCache.find(name);
}

void OptixResourceManager::destroy(CudaBuffer *buffer)
{
    SYRINX_EXPECT(buffer);
    mBufferCache.remove(buffer->getName());
}


void OptixResourceManager::destroy(CudaTexture *texture)
{
    SYRINX_EXPECT(texture);
    mTextureCache.remove(texture->getName());
}

} // namespace Syrinx
