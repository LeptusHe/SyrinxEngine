uniformBuffer = {
    type = "uniform-buffer",
    value = {
        int_parameters = 1,
        float_parameter = 1.0,
        light_info = {
            type = point_light,
            intensity = 100
        }
    }
}


--[[gbuffer_shader = {
    file_name = "pass-name",
    vertex_program_parameters = {
        diffuse_tex = {
            type = texture2D,
            value = {
                file = "image-file-name",
                format = "rgb8"
            }
        },

        uniformBuffer = {
            type = "uniform-buffer",
            value = {
                int_parameters = 1,
                float_parameter = 1,
                light_info = {
                    type = point_light,
                    intensity = 100
                }
            }
        }
    },

    fragment_program_parameters = {

    }
}

material = {
    gbuffer_shader,
    ao_shader
}
--]]

