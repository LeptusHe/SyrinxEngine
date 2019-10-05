shader = {
    vertex_program = {
        file = "syrinx-default-vs.vert"
    },
    fragment_program = {
        file = "syrinx-display-fragment-attribute.frag",
        predefined_macros = {
            { macro = "SYRINX_DISPLAY_WORLD_POSITION", value = ""}
        }
    }
}
