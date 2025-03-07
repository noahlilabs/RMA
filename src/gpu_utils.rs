use anyhow::Result;
use wgpu::util::DeviceExt;

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("No suitable GPU adapters found."))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("InfiniAttention Device"),
                    required_features: wgpu::Features::default(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await?;
        Ok(Self { device, queue })
    }
}

// ----------------------------------------------------------------
// The rest of this file remains the same. 
// Just ensure bytemuck references compile now that it's included.
// ----------------------------------------------------------------

use bytemuck;

pub fn create_storage_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    data: &[T],
    label: &str,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

pub fn create_empty_storage_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    len: usize,
    label: &str,
) -> wgpu::Buffer {
    let size = (len * std::mem::size_of::<T>()) as u64;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

pub async fn download_buffer<T: bytemuck::Pod>(
    context: &GpuContext,
    buffer: &wgpu::Buffer,
    len: usize,
) -> Result<Vec<T>> {
    let size = (len * std::mem::size_of::<T>()) as u64;
    let mut encoder = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Download Encoder"),
    });
    let staging = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    context.queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
    context.device.poll(wgpu::Maintain::Wait);
    rx.receive().await;

    let data = slice.get_mapped_range();
    let result = bytemuck::cast_slice(&data).to_vec();
    Ok(result)
}