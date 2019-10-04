#include "SyrinxAccelerateStructure.h"
#include <Component/SyrinxRenderer.h>
#include <Logging/SyrinxLogManager.h>
#include "SyrinxCudaBuffer.h"

namespace Syrinx {


void AccelerateStructure::build(const std::vector<Entity*>& entityList)
{
	std::vector<Entity*> entityListWithRenderer;
	for (const auto entity : entityList) {
        SYRINX_ASSERT(entity);
        if (!entity->hasComponent<Renderer>()) {
            SYRINX_WARN_FMT("fail to create accelerate structure for entity [{}] because it has not renderer component",
                            entity->getName());
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
	for (size_t i = 0; i < entityCount; ++ i) {
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

	OptixAccelBufferSizes bufferSize;
}

}