<shader name="lighting/pbr">
    <input-parameter-set>
        <parameter name="diffuseTex" type="texture-2d" value="white">
            <scale>0.1 0.1</scale>
            <offset>0.1 0.2</offset>
        </parameter>
        <parameter name="skyCube" type="texture-cube" value="white"/>
        <parameter name="skyColor" type="color" value="1.0 1.0 0.0 1.0"/>
        <parameter name="texIndex" type="int" value="1"/>
        <parameter name="modelScale" type="float" value="10.0"/>
    </input-parameter-set>

    <pass name="shadow">
        <vertex-program>
            <input-vertex-attribute-set>
                <attribute name="inPos" semantic="position" data-type="float3"/>
            </input-vertex-attribute-set>
            <code-file>../shadow_vertex.glsl</code-file>
        </vertex-program>
        <fragment-program>
            <code-file>../shadow_fragment.glsl</code-file>
        </fragment-program>
    </pass>

    <pass name="gbuffer">
        <vertex-program>
            <input-vertex-attribute-set>
                <attribute name="inPos" semantic="position" data-type="float3"/>
                <attribute name="inNormal" semantic="normal" data-type="float3"/>
                <attribute name="inTexCoord" semantic="texCoord" data-type="float2"/>
            </input-vertex-attribute-set>
            <input-parameter-set>
                <parameter ref="diffuseTex"/>
                <parameter ref="modelScale"/>
            </input-parameter-set>
            <code-file>../vertex.glsl</code-file>
        </vertex-program>
        <fragment-program>
            <input-parameter-set>
                <parameter ref="diffuseTex"/>
                <parameter ref="skyColor"/>
            </input-parameter-set>
            <code-file>../fragment.glsl</code-file>
        </fragment-program>
    </pass>
</shader>

