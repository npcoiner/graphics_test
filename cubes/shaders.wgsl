
struct VPUniforms {
    vpMatrix: mat4x4<f32>
};

@group(0) @binding(0) var<uniform> vp : VPUniforms;
@group(0) @binding(1) var<storage, read> colorBuffer : array<vec3<f32>>;

struct VertexInput {
    @location(0) position : vec3<f32>,
    @location(1) instancePos : vec3<f32>,
    @builtin(instance_index) instanceIndex : u32
};

struct VertexOutput {
    @builtin(position) pos : vec4<f32>,
    @location(0) color : vec3<f32>
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out : VertexOutput;
    let modelPos = input.position + input.instancePos;
    out.pos = vp.vpMatrix * vec4<f32>(modelPos, 1.0);
    out.color = colorBuffer[input.instanceIndex];
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 1.0);
}


