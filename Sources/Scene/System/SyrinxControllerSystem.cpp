#include "System/SyrinxControllerSystem.h"
#include "Component/SyrinxController.h"

namespace Syrinx {

void ControllerSystem::update(entityx::EntityManager& entityManager, entityx::EventManager& eventManager, entityx::TimeDelta timeDelta)
{
    entityx::ComponentHandle<Syrinx::Controller*> controllerHandle;
    for (entityx::Entity entity : entityManager.entities_with_components(controllerHandle)) {
        auto& controller = *controllerHandle;
        controller->update(timeDelta);
    }
}

} // namespace Syrinx
