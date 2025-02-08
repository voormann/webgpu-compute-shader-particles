const canvas = document.querySelector('canvas');
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
const ctx = canvas.getContext('webgpu');
const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

canvas.width = canvas.clientWidth;
canvas.height = canvas.clientHeight;
ctx.configure({
    device: device,
    format: presentationFormat,
    alphaMode: 'opaque'
});

let lastTime = 0;

const numParticles = 1000000;
const particleData = new Float32Array(numParticles * 4);
for (let i = 0; i < numParticles; i++) {
    particleData[i * 4]     = Math.random(); // x
    particleData[i * 4 + 1] = Math.random(); // y
    particleData[i * 4 + 2] = (Math.random() - 0.5) * 0.1; // vx
    particleData[i * 4 + 3] = (Math.random() - 0.5) * 0.1; // vy
}
const particleBuffer = device.createBuffer({
    size: particleData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true
});
new Float32Array(particleBuffer.getMappedRange()).set(particleData);
particleBuffer.unmap();

const uniformData = new Float32Array(4); // delta time, mouse click state, mouseX, mouseY
const uniformBuffer = device.createBuffer({
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});

const computeModule = device.createShaderModule({
    code: `
struct Particle {
    pos : vec2<f32>,
    vel : vec2<f32>,
};

struct Particles {
    particles : array<Particle>,
};

@group(0) @binding(0) var<storage, read_write> particles : Particles;

struct Uniforms {
    dt : f32,
    mouseActive : f32,
    mouse : vec2<f32>,
};

@group(0) @binding(1) var<uniform> uniforms : Uniforms;

fn random(st: vec2<f32>) -> f32 {
    return fract(sin(dot(st, vec2<f32>(12.9898, 78.233))) * 43758.5453123);
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    let index = GlobalInvocationID.x;

    if (index >= arrayLength(&particles.particles)) {
        return;
    }

    var p = particles.particles[index];
    let dir = uniforms.mouse - p.pos;
    let dist = dot(dir, dir);

    if (dist < 0.04) {
        if (u32(uniforms.mouseActive) == 1u) {
            p.vel += normalize(dir) * uniforms.dt;
            let tangentialForce = vec2(-dir.y, dir.x) * 0.2 / sqrt(dist);
            p.vel += tangentialForce * uniforms.dt;
        } else if (u32(uniforms.mouseActive) == 2u) {
            p.vel -= normalize(dir) * uniforms.dt;
        }
    }

    let noise = vec2(random(p.pos), random(p.vel)) * 2.0 - 1.0;

    p.vel += noise * 0.1 * uniforms.dt;
    p.vel *= 0.99;
    p.pos += p.vel * uniforms.dt;

    if (p.pos.x < 0.0 || p.pos.x > 1.0) {
        p.vel.x = -p.vel.x;
        p.pos.x = clamp(p.pos.x, 0.0, 1.0);
    }

    if (p.pos.y < 0.0 || p.pos.y > 1.0) {
        p.vel.y = -p.vel.y;
        p.pos.y = clamp(p.pos.y, 0.0, 1.0);
    }

    particles.particles[index] = p;
}
`
});
const computePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: computeModule, entryPoint: 'cs_main' }
});
const computeBindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
        { binding: 0, resource: { buffer: particleBuffer } },
        { binding: 1, resource: { buffer: uniformBuffer } }
    ]
});
const renderShaderModule = device.createShaderModule({
    code: `
struct Particle {
    pos : vec2<f32>,
    vel : vec2<f32>,
};

struct Particles {
    particles : array<Particle>,
};

@group(0) @binding(0) var<storage, read> particles : Particles;

struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) color : vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) VertexIndex : u32) -> VertexOutput {
    let p = particles.particles[VertexIndex];
    var output : VertexOutput;

    output.Position = vec4<f32>(p.pos * 2.0 - vec2<f32>(1.0, 1.0), 0.0, 1.0);

    let teal = vec3<f32>(0.149, 1.0, 0.875);
    let ember = vec3<f32>(1.0, 0.173, 0.063);

    output.color = vec4<f32>(mix(teal, ember, p.vel.x + p.vel.y * 2.0), 1.0);

    return output;
}

@fragment
fn fs_main(@location(0) color : vec4<f32>) -> @location(0) vec4<f32> {
    return color;
}
`
});
const renderPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: renderShaderModule, entryPoint: 'vs_main' },
    fragment: { module: renderShaderModule, entryPoint: 'fs_main', targets: [{ format: presentationFormat }] },
    primitive: { topology: 'point-list' }
});
const renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
        { binding: 0, resource: { buffer: particleBuffer } }
    ]
});

function render(now) {
    uniformData[0] = Math.min(now - lastTime, 30) * 0.001;
    lastTime = now;
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const commandEncoder = device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, computeBindGroup);
    computePass.dispatchWorkgroups(Math.ceil(numParticles / 64));
    computePass.end();

    const textureView = ctx.getCurrentTexture().createView();
    const renderPass = commandEncoder.beginRenderPass({
        colorAttachments: [{
            view: textureView,
            clearValue: [0, 0, 0, 1],
            loadOp: 'clear',
            storeOp: 'store'
        }]
    });
    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, renderBindGroup);
    renderPass.draw(numParticles, 1, 0, 0);
    renderPass.end();
    device.queue.submit([commandEncoder.finish()]);

    requestAnimationFrame(render);
}

requestAnimationFrame(render);

canvas.addEventListener('mousemove', (event) => {
    uniformData[2] = event.clientX / innerWidth;
    uniformData[3] = 1 - event.clientY / innerHeight;
});

canvas.addEventListener('mousedown', (event) => {
    uniformData[1] = event.button === 0 ? 1 : 2;
});

canvas.addEventListener('mouseup', () => {
    uniformData[1] = 0;
});

canvas.addEventListener('contextmenu', (event) => {
    event.preventDefault();
});

canvas.addEventListener('touchmove', (event) => {
    uniformData[2] = event.touches[0].clientX / innerWidth;
    uniformData[3] = 1 - event.touches[0].clientY / innerHeight;
    event.preventDefault();
});

canvas.addEventListener('touchstart', () => {
    uniformData[1] = event.touches.length === 2 ? 2 : 1;
});

canvas.addEventListener('touchend', () => {
    uniformData[1] = 0;
});

window.addEventListener('resize', () => {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
});