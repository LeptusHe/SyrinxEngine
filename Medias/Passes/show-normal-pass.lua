function render(renderContext, cameraList, scene)
    renderContext:clearRenderTarget(nil, syrinx.Color.new(0.0, 0.0, 1.0, 1.0))
    renderContext:clearDepth(nil, 1.0)

    if scene == nil then
        print("scene is null")
        return
    end

    local camera = cameraList[1]
    if camera == nil then
        print("camera is null")
    end

    renderState = syrinx.RenderState.new()
    renderState.viewportState.viewport = syrinx.createViewport(0, 0, 800, 800)

    renderContext:pushRenderState()
    renderContext:setRenderState(renderState)
    local entityList = scene:getEntitiesWithRenderer()
    for index, entity in ipairs(entityList) do
        syrinx.entityRenderer:render(camera, renderContext, entity, "display-world-normal.shader")
    end
end