<shader name="unlit/constant-color">
    <input-parameter-set>
        <parameter name="displayColor" type="Color" value="1.0 0.0 0.0 1.0"/>
    </input-parameter-set>

    <pass name="lighting">
        <vertex-program>
            <input-vertex-attribute-set>
                <attribute name="inPos" semantic="position" data-type="float3"/>
            </input-vertex-attribute-set>
            <code-file>constant-color-vs.glsl</code-file>
        </vertex-program>
        <fragment-program>
            <input-parameter-set>
                <parameter ref="displayColor"/>
            </input-parameter-set>
            <code-file>constant-color-fs.glsl</code-file>
        </fragment-program>
    </pass>
</shader>

