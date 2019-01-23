<shader name="lighting/bump">
    <input-parameter-set>
        <parameter name="uAlbedoMap" type="texture-2d" value="white">
            <scale>1.0 1.0</scale>
            <offset>0.0 0.0</offset>
        </parameter>
        <parameter name="uNormalMap" type="texture-2d" value="white">
            <scale>1.0 1.0</scale>
            <offset>0.0 0.0</offset>
        </parameter>
    </input-parameter-set>

    <pass name="lighting">
        <vertex-program>
            <input-vertex-attribute-set>
                <attribute name="inPos" semantic="position" data-type="float3"/>
                <attribute name="inNormal" semantic="normal" data-type="float3"/>
            </input-vertex-attribute-set>
            <code-file>bump-vs.glsl</code-file>
        </vertex-program>
        <fragment-program>
            <input-parameter-set>
                <parameter ref="uAlbedoMap"/>
                <parameter ref="uNormalMap"/>
            </input-parameter-set>
            <code-file>bump-fs.glsl</code-file>
        </fragment-program>
    </pass>
</shader>

