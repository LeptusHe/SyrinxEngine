#include "SyrinxOptixContext.h"
#include <cuda_runtime.h>
#include <optix_stubs.h>
#include <Exception/SyrinxException.h>
#include <Logging/SyrinxLogManager.h>
#include <Scene/SyrinxEntity.h>
#include <Component/SyrinxRenderer.h>
#include "SyrinxRadianceAssert.h"
#include "SyrinxCudaBuffer.h"

namespace Syrinx {

void OptixLogCallback(unsigned int level, const char *tag, const char *message, void *callbackData)
{
    const char* format = "Optix Log [tag={}] [message={}]";
    switch (level) {
        case 1: SYRINX_FAULT_FMT(format, tag, message); break;
        case 2: SYRINX_ERROR_FMT(format, tag, message); break;
        case 3: SYRINX_WARN_FMT(format, tag, message); break;
        case 4: SYRINX_INFO_FMT(format, tag, message); break;
        default: SYRINX_FAULT_FMT(format, tag, message); break;
    }
}




OptixContext::~OptixContext()
{

}


void OptixContext::init()
{
    cudaFree(nullptr);

    int numDevices;
    cudaGetDeviceCount(&numDevices);

    if (numDevices == 0) {
        SYRINX_THROW_EXCEPTION(ExceptionCode::CUDAError,
            "fail to init OptiX context because no CUDA capable devices found");
    }
    SYRINX_OPTIX_ASSERT(optixInit());
    SYRINX_INFO("succeed to init OptiX context");

    const int deviceId = 0;
    SYRINX_CUDA_ASSERT(cudaSetDevice(deviceId));
    SYRINX_CUDA_ASSERT(cudaStreamCreate(&mCudaStream));

    cudaGetDeviceProperties(&mCudaDeviceProperties, deviceId);
    SYRINX_INFO_FMT("running on device [{}]", mCudaDeviceProperties.name);

    auto result = cuCtxGetCurrent(&mCudaContext);
    if (result != CUDA_SUCCESS) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::CUDAError,
            "fail to query current context with error code [{}]", result);
    }

    SYRINX_OPTIX_ASSERT(optixDeviceContextCreate(mCudaContext, nullptr, &mOptixContext));
    SYRINX_OPTIX_ASSERT(optixDeviceContextSetLogCallback(mOptixContext, OptixLogCallback, nullptr, 4));
}


AccelerateStructure OptixContext::buildAccelerateStructure(const std::vector<Entity*>& entityList)
{
    std::vector<Entity*> entityListWithRenderer;
    for (const auto entity : entityList) {
        SYRINX_ASSERT(entity);
        if (!entity->hasComponent<Renderer>()) {
            SYRINX_WARN_FMT("fail to create accelerate structure for entity [{}] because it has not renderer component", entity->getName());
            continue;
        }
        entityListWithRenderer.push_back(entity);
    }

    const auto entityCount = entityListWithRenderer.size();
    std::vector<CudaBuffer> vertexBufferList(entityCount);
    std::vector<CudaBuffer> normalBufferList(entityCount);
    std::vector<CudaBuffer> tangentBufferList(entityCount);
    std::vector<CudaBuffer> texCoordBufferList(entityCount);
    std::vector<CudaBuffer> indexBufferList(entityCount);

    std::vector<CUdeviceptr> vertexBufferMemoryList(entityCount);
    std::vector<OptixBuildInput> buildInputList(entityCount);
    for (size_t i = 0; i < entityCount; ++i) {
        const auto& entity = entityListWithRenderer[i];
        const auto& renderer = entity->getComponent<Renderer>();
        auto mesh = renderer.getMesh();
        SYRINX_ASSERT(mesh);

        const auto vertexCount = mesh->getNumVertex();
        vertexBufferList[i].allocateAndUpload(mesh->getPositionSet(), vertexCount);
        normalBufferList[i].allocateAndUpload(mesh->getNormalSet(), vertexCount);
        tangentBufferList[i].allocateAndUpload(mesh->getTangentSet(), vertexCount);
        indexBufferList[i].allocateAndUpload(mesh->getIndexSet(), vertexCount);

        const auto uvChannel = mesh->getUVChannel(0);
        SYRINX_ASSERT(uvChannel);
        texCoordBufferList[i].allocateAndUpload(uvChannel->uvSet, uvChannel->numElement * vertexCount);

        vertexBufferMemoryList[i] = vertexBufferList[i].getDevicePtr();

        auto& buildInput = buildInputList[i];
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        auto& triangleArray = buildInput.triangleArray;
        triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleArray.vertexStrideInBytes = sizeof(Point3f);
        triangleArray.numVertices = vertexCount;
        triangleArray.vertexBuffers = &(vertexBufferMemoryList[i]);

        static_assert(sizeof(unsigned int) == sizeof(uint32_t));
        triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleArray.indexStrideInBytes = 3 * sizeof(*mesh->getIndexSet());
        triangleArray.numIndexTriplets = mesh->getNumTriangle();
        triangleArray.indexBuffer = indexBufferList[i].getDevicePtr();

        unsigned int flag = 0;
        triangleArray.flags = &flag;
        triangleArray.numSbtRecords = 1;
        triangleArray.sbtIndexOffsetBuffer = 0;
        triangleArray.sbtIndexOffsetSizeInBytes = 0;
        triangleArray.sbtIndexOffsetStrideInBytes = 0;
    }

    OptixAccelBuildOptions buildOptions;
    buildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    buildOptions.motionOptions.numKeys = 1;
    buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes accelerateStructBufferSizes;
    SYRINX_OPTIX_ASSERT(optixAccelComputeMemoryUsage(mOptixContext,
                                                             &buildOptions,
                                                             buildInputList.data(),
                                                             buildInputList.size(),
                                                             &accelerateStructBufferSizes));

    CudaBuffer tempBuffer;
    tempBuffer.allocate(accelerateStructBufferSizes.tempSizeInBytes);

    CudaBuffer outputBuffer;
    outputBuffer.allocate(accelerateStructBufferSizes.outputSizeInBytes);

    CudaBuffer compactedSizeBuffer;
    compactedSizeBuffer.allocate(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.getDevicePtr();

    OptixTraversableHandle traversableHandle;
    SYRINX_OPTIX_ASSERT(optixAccelBuild(mOptixContext,
                                                nullptr,
                                                &buildOptions,
                                                buildInputList.data(),
                                                buildInputList.size(),
                                                tempBuffer.getDevicePtr(),
                                                tempBuffer.getSize(),
                                                outputBuffer.getDevicePtr(),
                                                outputBuffer.getSize(),
                                                &traversableHandle,
                                                &emitDesc,
                                                1));
    cudaDeviceSynchronize();
    SYRINX_CUDA_ASSERT(cudaGetLastError());

    uint64_t compactedSize = 0;
    compactedSizeBuffer.download(&compactedSize, 1);

    CudaBuffer accelerateStructBuffer;
    accelerateStructBuffer.allocate(compactedSize);
    SYRINX_OPTIX_ASSERT(optixAccelCompact(mOptixContext,
                                                  nullptr,
                                                  traversableHandle,
                                                  accelerateStructBuffer.getDevicePtr(),
                                                  accelerateStructBuffer.getSize(),
                                                  &traversableHandle))
    cudaDeviceSynchronize();
    SYRINX_CUDA_ASSERT(cudaGetLastError());


}


} // namespace Syrinx
