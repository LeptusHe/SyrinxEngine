pipeline = {
    file = "program.ptx",
    program_groups = {
        program_groups = {
            type = "ray-gen",

        },
        program_groups = {
            type = "hit-group",
            entry_functions = {
                closest_hit = "closest_hit_entry",
                any_hit = "any_hit_entry"
            }
        },
        program_groups = {
            type = "miss"
        },
        program_groups = {
            type = "exception",
        }
    }

}