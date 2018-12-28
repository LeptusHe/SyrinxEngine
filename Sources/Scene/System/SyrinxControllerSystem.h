#pragma once
#include <entityx/entityx.h>

namespace Syrinx {

class ControllerSystem : public entityx::System<ControllerSystem> {
public:
    void update(entityx::EntityManager& entityManager, entityx::EventManager& eventManager, entityx::TimeDelta timeDelta) override;
};

} // namespace Syrinx


