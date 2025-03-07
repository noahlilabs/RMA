use anyhow::Result;
use clap::Parser;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::Arc;

use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use futures::executor::block_on;

mod file_reader;
mod tokenizer;
mod gpu_utils;
mod infini_attention_gpu;

use file_reader::convert_to_text;
use tokenizer::tokenize;
use gpu_utils::GpuContext;
use infini_attention_gpu::InfiniAttentionGpu;

/// Command-line arguments
#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Extended Infini-Attention Demo (txt/pdf/docx) with advanced GPU pipeline",
    long_about = None
)]
struct Args {
    /// Path to a text, PDF, or DOCX file.
    #[arg(short, long)]
    input: PathBuf,

    /// Segment size
    #[arg(short, long, default_value_t = 16)]
    segment_size: usize,

    /// Embedding dimension
    #[arg(short, long, default_value_t = 12)]
    embed_dim: usize,

    /// Number of tokens in the random vocab
    #[arg(short, long, default_value_t = 10000)]
    vocab_size: usize,

    /// Number of heads
    #[arg(long, default_value_t = 1)]
    heads: usize,

    /// Use GPU
    #[arg(long, default_value_t = false)]
    gpu: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // 1) Possibly create a GPU context
    let gpu_ctx = if args.gpu {
        println!("Initializing GPU context...");
        Some(block_on(GpuContext::new())?)
    } else {
        None
    };

    // 2) Convert file to text
    let reader = convert_to_text(&args.input)?;

    // 3) Build (GPU-based) InfiniAttention
    // or fallback to CPU if `gpu: false` (in which case you'd have a CPU-based version).
    let d_model = args.embed_dim;
    assert!(d_model % 3 == 0, "embed_dim must be multiple of 3");
    let mut infini_gpu = if let Some(gpu) = gpu_ctx {
        Some(InfiniAttentionGpu::new(Arc::new(gpu), args.heads, d_model / 3, d_model / 3, d_model))
    } else {
        None
    };

    // 4) Create random embedding table on CPU for demonstration
    let embedding_table =
        Array2::<f32>::random((args.vocab_size, d_model), Uniform::new(-0.1, 0.1));

    let seg_size = args.segment_size;
    let mut token_buffer = Vec::new();
    let mut global_sum = Array1::<f32>::zeros(d_model);
    let mut global_count = 0usize;

    // 5) Read lines -> tokenize -> buffer
    for line_result in BufReader::new(reader).lines() {
        let line = line_result?;
        let tokens = tokenize(&line);
        for t in tokens {
            token_buffer.push(t);
            if token_buffer.len() >= seg_size {
                // Process segment
                let seg = &token_buffer[..seg_size];
                let output = if let Some(ref mut infini) = infini_gpu {
                    // GPU-based approach
                    block_on(process_segment_gpu(seg, &embedding_table, infini))?
                } else {
                    // CPU fallback (not shown in detail here)
                    process_segment_cpu(seg, &embedding_table, d_model)
                };
                // Accumulate for global average
                for row_idx in 0..seg_size {
                    for c in 0..d_model {
                        global_sum[c] += output[row_idx * d_model + c];
                    }
                    global_count += 1;
                }
                token_buffer.drain(0..seg_size);
            }
        }
    }

    // leftover
    if !token_buffer.is_empty() {
        let seg = &token_buffer[..];
        let output = if let Some(ref mut infini) = infini_gpu {
            block_on(process_segment_gpu(seg, &embedding_table, infini))?
        } else {
            process_segment_cpu(seg, &embedding_table, d_model)
        };
        let n_left = seg.len();
        for row_idx in 0..n_left {
            for c in 0..d_model {
                global_sum[c] += output[row_idx * d_model + c];
            }
            global_count += 1;
        }
    }

    if global_count > 0 {
        let avg = global_sum.mapv(|x| x / (global_count as f32));
        println!("Processed {} tokens, final avg of first 10 dims:", global_count);
        for i in 0..10.min(d_model) {
            print!("{:.4} ", avg[i]);
        }
        println!();
    } else {
        println!("No tokens processed.");
    }

    Ok(())
}

async fn process_segment_gpu(
    tokens: &[usize],
    embedding_table: &Array2<f32>,
    infini: &mut InfiniAttentionGpu,
) -> Result<Vec<f32>> {
    let d_model = infini.d_model;
    let n = tokens.len();

    // Build the (N x d_model) embeddings on CPU
    // Then we pass that to the GPU in forward()
    let mut x_seg = vec![0.0_f32; n * d_model];
    for i in 0..n {
        let tok_id = tokens[i] % embedding_table.len_of(Axis(0));
        for c in 0..d_model {
            x_seg[i * d_model + c] = embedding_table[[tok_id, c]];
        }
    }

    // forward on GPU
    let out = infini.forward(&x_seg, n).await?;
    Ok(out)
}

fn process_segment_cpu(
    tokens: &[usize],
    embedding_table: &Array2<f32>,
    d_model: usize,
) -> Vec<f32> {
    // Just do a naive pass or something. Not shown in detail.
    let n = tokens.len();
    let mut x_seg = vec![0.0; n * d_model];
    for (i, &tok_id) in tokens.iter().enumerate() {
        let row_idx = tok_id % embedding_table.len_of(Axis(0));
        for c in 0..d_model {
            x_seg[i * d_model + c] = embedding_table[[row_idx, c]];
        }
    }
    // ...some CPU-based attention...
    x_seg // return the same for demo
}