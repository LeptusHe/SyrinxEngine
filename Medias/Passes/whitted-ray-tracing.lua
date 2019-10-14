pipeline = {
    ray_gen_program_groups = {
        file = "file-name",
        entry_function = "__raygen__render"
    },
    hit_program_groups_array = {
        {
            ray_type = "radiance",
            closest_hit_program = {file = "radiance.ptx", entry_function = "closest_hit_entry"},
            any_hit_program = {file = "radiance.ptx", entry_function = "closest_hit_entry"},
            miss_hit_program = {file = "radiance.ptx", entry_function = "closest_hit_entry"},
        },
        {
            ray_type = "shaow",
            closest_hit_program = {file = "shadow.ptx", entry_function = "shadow_closest_hit"}
        }
    },
    exception_program_group = {
        file = "exception.ptx",
        entry_function = "exception"
    },
    miss_program_groups_array = {
        {},
        {}
    },
    callable_program_group_array = {
        {
            direct_callable = {}
        },
        {

        }
    }
}