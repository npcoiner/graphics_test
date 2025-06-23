struct VPUniforms {
	vpMatrix: mat4x4<f32>
};

struct ColorInstance {
    rgbaOffset : vec4<f32> // xyz = color, w = time offset
};

struct SimParams {
    time : f32,
    count : u32
};

@group(0) @binding(0) var<uniform> vp : VPUniforms;
@group(1) @binding(0) var<storage, read_write> colors : array<ColorInstance>;
@group(1) @binding(1) var<uniform> sim : SimParams;

struct VertexInput {
    @location(0) position : vec3<f32>,
    @location(1) instancePos : vec3<f32>,
    @location(2) color : vec4<f32>,
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
    out.color = input.color.xyz;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 1.0);
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sim.count) { return; }
    var c = colors[i].rgbaOffset;
    let t = sim.time + c.w;
    c.x = 0.5 + 0.5 * sin(t);
    c.y = 0.5 + 0.5 * sin(t + 2.094); // offset phases
    c.z = 0.5 + 0.5 * sin(t + 4.188);
    colors[i].rgbaOffset = c;
}
