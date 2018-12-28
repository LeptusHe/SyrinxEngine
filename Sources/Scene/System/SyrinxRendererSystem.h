#pragma once
#include <entityx/entityx.h>
#include "Component/SyrinxRenderer.h"

namespace Syrinx {

class RendererSystem : public entityx::System<RendererSystem> {
public:
    void update(entityx::EntityManager& entityManager, entityx::EventManager& eventManager, entityx::TimeDelta timeDelta) override;
};

} // namespace Syrinx