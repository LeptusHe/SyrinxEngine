<shader name="unlit/display-vertex-normal">
    <pass name="lighting">
        <vertex-program>
            <input-vertex-attribute-set>
                <attribute name="inPos" semantic="position" data-type="float3"/>
                <attribute name="inNormal" semantic="normal" data-type="float3"/>
            </input-vertex-attribute-set>
            <code-file>display-vertex-normal.vs</code-file>
        </vertex-program>
        <fragment-program>
            <code-file>display-vertex-normal.fs</code-file>
        </fragment-program>
    </pass>
</shader>

