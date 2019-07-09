function create_render_technique()
    render_technique = {
        name = "auto-exposure",
        pass = {
            calculate_average_luminance,
            tone_mapping
        }
    }
    return render_technique
end

