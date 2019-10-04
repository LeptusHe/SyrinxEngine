#pragma once
#include <vector>
#include <optix.h>
#include <Scene/SyrinxEntity.h>

namespace Syrinx {

class AccelerateStructure {
public:
	AccelerateStructure() = default;
	~AccelerateStructure() = default;
	void build(const std::vector<Entity*>& entityList);
	
private:
	OptixTraversableHandle mHandle = 0;
};	
	
} // namespace Syrinx::Radiance