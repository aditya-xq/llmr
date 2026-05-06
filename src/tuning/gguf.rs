use crate::tuning::{GgufFacts, OptimizeError};
use std::io::Read;
use std::path::Path;

pub struct GgufParser;

impl GgufParser {
    pub fn extract(path: &Path) -> Result<GgufFacts, OptimizeError> {
        let file = std::fs::File::open(path).map_err(OptimizeError::Io)?;
        let metadata = file.metadata().map_err(OptimizeError::Io)?;
        let mut reader = std::io::BufReader::new(file);
        extract_from_reader(&mut reader, path, metadata.len())
    }
}

pub fn extract_from_reader<R: Read + ?Sized>(
    reader: &mut R,
    path: &Path,
    file_size: u64,
) -> Result<GgufFacts, OptimizeError> {
    let magic = {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf).map_err(OptimizeError::Io)?;
        u32::from_le_bytes(buf)
    };

    if magic != 0x46554747 {
        return Err(OptimizeError::Parse("Not a valid GGUF file".to_string()));
    }

    let version = {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf).map_err(OptimizeError::Io)?;
        u32::from_le_bytes(buf)
    };

    if version < 3 {
        return Err(OptimizeError::Parse(format!(
            "Unsupported GGUF version: {}",
            version
        )));
    }

    let tensor_count = read_u64(reader)?;
    let metadata_kv_count = read_u64(reader)?;

    let mut architecture = None;
    let mut context_length = None;
    let mut embedding_length = None;
    let mut block_count = None;
    let mut feed_forward_length = None;
    let mut attention_head_count = None;
    let mut attention_head_count_kv = None;
    let mut rope_dimension_count = None;
    let mut rope_scaling_type = None;
    let mut rope_scaling_factor = None;
    let mut rope_scaling_original_context_length = None;
    let mut chat_template = None;
    let mut model_name = None;
    let mut quantization_version = None;
    let mut file_type = None;
    let mut alignment = None;

    for _ in 0..metadata_kv_count {
        let key = read_string(reader)?;
        let value_type = read_u32(reader)? as u8;

        match key.as_str() {
            "general.architecture" => {
                if value_type == 8 {
                    architecture = Some(read_string(reader)?);
                }
            }
            "general.name" => {
                if value_type == 8 {
                    model_name = Some(read_string(reader)?);
                }
            }
            "general.alignment" => {
                if value_type == 4 {
                    alignment = Some(read_u32(reader)?);
                }
            }
            "general.quantization_version" => {
                if value_type == 4 {
                    quantization_version = Some(read_u32(reader)?);
                }
            }
            "general.file_type" => {
                if value_type == 4 {
                    file_type = Some(read_u32(reader)?);
                }
            }
            "llama.context_length" => {
                if value_type == 4 {
                    context_length = Some(read_u32(reader)?);
                }
            }
            "llama.embedding_length" => {
                if value_type == 4 {
                    embedding_length = Some(read_u32(reader)?);
                }
            }
            "llama.block_count" => {
                if value_type == 4 {
                    block_count = Some(read_u32(reader)?);
                }
            }
            "llama.feed_forward_length" => {
                if value_type == 4 {
                    feed_forward_length = Some(read_u32(reader)?);
                }
            }
            "llama.attention.head_count" => {
                if value_type == 4 {
                    attention_head_count = Some(read_u32(reader)?);
                }
            }
            "llama.attention.head_count_kv" => {
                if value_type == 4 {
                    attention_head_count_kv = Some(read_u32(reader)?);
                }
            }
            "llama.rope.dimension_count" => {
                if value_type == 4 {
                    rope_dimension_count = Some(read_u32(reader)?);
                }
            }
            "llama.rope.scaling.factor" => {
                if value_type == 6 {
                    rope_scaling_factor = Some(read_f32(reader)?);
                }
            }
            "llama.rope.scale_linear" => {
                if value_type == 6 {
                    let scale = read_f32(reader)?;
                    if rope_scaling_factor.is_none() {
                        rope_scaling_factor = Some(scale);
                    }
                }
            }
            "llama.rope.scaling.type" => {
                if value_type == 8 {
                    rope_scaling_type = Some(read_string(reader)?);
                }
            }
            "llama.rope.scaling.original_context_length" => {
                if value_type == 4 {
                    rope_scaling_original_context_length = Some(read_u32(reader)?);
                }
            }
            "llama.rope.scaling.freq_base" => {
                skip_value(reader, value_type)?;
            }
            "llama.rope.scale"
            | "llama.rope.scale.linear"
            | "llama.rope.scale Yarn"
            | "llama.rope.scaling.original_context" => {
                skip_value(reader, value_type)?;
            }
            "tokenizer.chat_template" => {
                if value_type == 8 {
                    chat_template = Some(read_string(reader)?);
                }
            }
            _ => {
                skip_value(reader, value_type)?;
            }
        }
    }

    let size_label = compute_size_label(file_size);

    Ok(GgufFacts {
        path: path.to_path_buf(),
        architecture: architecture.unwrap_or_else(|| "unknown".to_string()),
        model_name,
        size_label,
        quantization_version,
        file_type,
        alignment,
        context_length,
        embedding_length,
        block_count,
        feed_forward_length,
        attention_head_count,
        attention_head_count_kv,
        rope_dimension_count,
        rope_scaling_type,
        rope_scaling_factor,
        rope_scaling_original_context_length,
        chat_template,
        tensor_count: tensor_count as usize,
        weight_bytes: file_size,
    })
}

pub fn read_u64<R: Read + ?Sized>(reader: &mut R) -> Result<u64, OptimizeError> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf).map_err(OptimizeError::Io)?;
    Ok(u64::from_le_bytes(buf))
}

pub fn read_u32<R: Read + ?Sized>(reader: &mut R) -> Result<u32, OptimizeError> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf).map_err(OptimizeError::Io)?;
    Ok(u32::from_le_bytes(buf))
}

pub fn read_f32<R: Read + ?Sized>(reader: &mut R) -> Result<f32, OptimizeError> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf).map_err(OptimizeError::Io)?;
    Ok(f32::from_le_bytes(buf))
}

pub fn read_string<R: Read + ?Sized>(reader: &mut R) -> Result<String, OptimizeError> {
    let len = read_u64(reader)? as usize;
    if len > 50_000_000 {
        return Err(OptimizeError::Parse(format!("string too large: {}", len)));
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).map_err(OptimizeError::Io)?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}

pub fn skip_value<R: Read + ?Sized>(reader: &mut R, value_type: u8) -> Result<(), OptimizeError> {
    match value_type {
        0 => {
            read_u8(reader)?;
        }
        1 => {
            read_u8(reader)?;
        }
        2 => {
            read_u16(reader)?;
        }
        3 => {
            read_u16(reader)?;
        }
        4 => {
            read_u32(reader)?;
        }
        5 => {
            read_u32(reader)?;
        }
        6 => {
            read_f32(reader)?;
        }
        7 => {
            read_u8(reader)?;
        }
        8 => {
            let _ = read_string(reader)?;
        }
        9 => {
            let elem_type = read_u32(reader)? as u8;
            let len = read_u64(reader)?;
            for _ in 0..len {
                skip_value(reader, elem_type)?;
            }
        }
        10 => {
            read_u64(reader)?;
        }
        11 => {
            read_u64(reader)?;
        }
        12 => {
            read_f64(reader)?;
        }
        _ => {
            skip_value(reader, 0)?;
        }
    }
    Ok(())
}

pub fn read_u8<R: Read + ?Sized>(reader: &mut R) -> Result<u8, OptimizeError> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf).map_err(OptimizeError::Io)?;
    Ok(buf[0])
}

pub fn read_u16<R: Read + ?Sized>(reader: &mut R) -> Result<u16, OptimizeError> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf).map_err(OptimizeError::Io)?;
    Ok(u16::from_le_bytes(buf))
}

pub fn read_f64<R: Read + ?Sized>(reader: &mut R) -> Result<f64, OptimizeError> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf).map_err(OptimizeError::Io)?;
    Ok(f64::from_le_bytes(buf))
}

fn compute_size_label(bytes: u64) -> Option<String> {
    let gb = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    if gb < 1.0 {
        Some(format!("{:.0}MB", bytes as f64 / (1024.0 * 1024.0)))
    } else if gb < 100.0 {
        Some(format!("{:.1}GB", gb))
    } else {
        Some(format!("{:.0}GB", gb))
    }
}
