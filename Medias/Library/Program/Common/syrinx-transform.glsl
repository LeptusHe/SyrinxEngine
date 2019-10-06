#ifndef SYRINX_TRANSFORM_PROGRAM
#define SYRINX_TRANSFORM_PROGRAM

vec4 SyrinxObjectToWorldPos(vec3 posObject)
{
    return SYRINX_MATRIX_WORLD * vec4(posObject, 1.0);
}


vec3 SyrinxObjectToWorldNormal(vec3 normalObject)
{
    mat3 normalMat = transpose(inverse(mat3(SYRINX_MATRIX_WORLD)));
    return normalize(normalMat * normalObject);
}


vec4 SyrinxObjectToClipPos(vec3 posObject)
{
    vec4 pos = vec4(posObject, 1.0);
    return SYRINX_MATRIX_PROJ * SYRINX_MATRIX_VIEW * SYRINX_MATRIX_WORLD * pos;
}

#endif // SYRINX_TRANSFORM_PROGRAM