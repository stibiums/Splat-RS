use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Read},
    path::Path,
};

use anyhow::{Context, Result, bail};
use glam::{Vec3, Vec4};
use thiserror::Error;

use crate::scene::{GaussianRaw, SplatScene, sigmoid};

#[derive(Clone, Copy, Debug, Default)]
pub struct LoadOptions {
    pub max_splats: Option<usize>,
    pub min_opacity: Option<f32>,
    pub max_world_scale: Option<f32>,
}

#[derive(Debug, Error)]
pub enum LoadError {
    #[error("missing required PLY property `{0}`")]
    MissingProperty(&'static str),
    #[error("unsupported PLY format `{0}`")]
    UnsupportedFormat(String),
    #[error("PLY file does not contain a vertex element")]
    MissingVertexElement,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PlyFormat {
    Ascii,
    BinaryLittleEndian,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScalarType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl ScalarType {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "char" | "int8" => Some(Self::I8),
            "uchar" | "uint8" => Some(Self::U8),
            "short" | "int16" => Some(Self::I16),
            "ushort" | "uint16" => Some(Self::U16),
            "int" | "int32" => Some(Self::I32),
            "uint" | "uint32" => Some(Self::U32),
            "float" | "float32" => Some(Self::F32),
            "double" | "float64" => Some(Self::F64),
            _ => None,
        }
    }

    fn read_f32(self, bytes: &[u8], cursor: &mut usize) -> Result<f32> {
        let value = match self {
            Self::I8 => read_array::<1>(bytes, cursor)?[0] as i8 as f32,
            Self::U8 => read_array::<1>(bytes, cursor)?[0] as f32,
            Self::I16 => i16::from_le_bytes(read_array(bytes, cursor)?) as f32,
            Self::U16 => u16::from_le_bytes(read_array(bytes, cursor)?) as f32,
            Self::I32 => i32::from_le_bytes(read_array(bytes, cursor)?) as f32,
            Self::U32 => u32::from_le_bytes(read_array(bytes, cursor)?) as f32,
            Self::F32 => f32::from_le_bytes(read_array(bytes, cursor)?),
            Self::F64 => f64::from_le_bytes(read_array(bytes, cursor)?) as f32,
        };
        Ok(value)
    }
}

#[derive(Clone, Debug)]
struct Property {
    name: String,
    scalar_type: ScalarType,
}

#[derive(Clone, Debug)]
struct PlyHeader {
    format: PlyFormat,
    vertex_count: usize,
    vertex_properties: Vec<Property>,
    data_start: usize,
}

pub fn load_scene(path: &Path, options: LoadOptions) -> Result<SplatScene> {
    let mut bytes = Vec::new();
    File::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read {}", path.display()))?;

    let header = parse_header(&bytes)?;
    let mut raw = match header.format {
        PlyFormat::Ascii => parse_ascii_vertices(&bytes[header.data_start..], &header)?,
        PlyFormat::BinaryLittleEndian => {
            parse_binary_vertices(&bytes[header.data_start..], &header)?
        }
    };

    raw = filter_splats(raw, options);

    if let Some(limit) = options.max_splats {
        raw = select_most_important(raw, limit);
    }

    if raw.is_empty() {
        bail!("{} did not contain any splats", path.display());
    }

    Ok(SplatScene::from_raw(raw, path.display().to_string()))
}

fn filter_splats(items: Vec<GaussianRaw>, options: LoadOptions) -> Vec<GaussianRaw> {
    if options.min_opacity.is_none() && options.max_world_scale.is_none() {
        return items;
    }

    let min_opacity = options.min_opacity.map(|value| value.clamp(0.0, 1.0));
    let max_world_scale = options.max_world_scale.map(|value| value.max(0.0));
    items
        .into_iter()
        .filter(|item| {
            if let Some(min_opacity) = min_opacity {
                if sigmoid(item.opacity_logit) < min_opacity {
                    return false;
                }
            }
            if let Some(max_world_scale) = max_world_scale {
                let max_scale = item.log_scale.exp().max_element();
                if max_scale > max_world_scale {
                    return false;
                }
            }
            true
        })
        .collect()
}

fn select_most_important(items: Vec<GaussianRaw>, limit: usize) -> Vec<GaussianRaw> {
    if limit >= items.len() {
        return items;
    }

    if limit == 0 {
        return Vec::new();
    }

    let mut ranked = items
        .into_iter()
        .enumerate()
        .map(|(index, item)| (index, gaussian_importance(&item), item))
        .collect::<Vec<_>>();

    ranked.select_nth_unstable_by(limit, |(_, importance_a, _), (_, importance_b, _)| {
        importance_b.total_cmp(importance_a)
    });
    ranked.truncate(limit);
    ranked.sort_unstable_by_key(|(index, _, _)| *index);
    ranked.into_iter().map(|(_, _, item)| item).collect()
}

fn gaussian_importance(raw: &GaussianRaw) -> f32 {
    raw.log_scale.element_sum().exp() * sigmoid(raw.opacity_logit)
}

fn parse_header(bytes: &[u8]) -> Result<PlyHeader> {
    let marker = b"end_header";
    let marker_start = bytes
        .windows(marker.len())
        .position(|window| window == marker)
        .context("PLY header is missing end_header")?;
    let mut data_start = marker_start + marker.len();
    while data_start < bytes.len() && (bytes[data_start] == b'\r' || bytes[data_start] == b'\n') {
        data_start += 1;
    }

    let header_text =
        std::str::from_utf8(&bytes[..marker_start]).context("PLY header is not valid UTF-8")?;
    let mut format = None;
    let mut vertex_count = None;
    let mut vertex_properties = Vec::new();
    let mut in_vertex = false;

    for line in header_text.lines() {
        let parts: Vec<_> = line.split_whitespace().collect();
        match parts.as_slice() {
            ["ply"] => {}
            ["format", "ascii", _] => format = Some(PlyFormat::Ascii),
            ["format", "binary_little_endian", _] => format = Some(PlyFormat::BinaryLittleEndian),
            ["format", other, _] => {
                return Err(LoadError::UnsupportedFormat((*other).into()).into());
            }
            ["element", "vertex", count] => {
                vertex_count = Some(count.parse::<usize>().context("invalid vertex count")?);
                in_vertex = true;
            }
            ["element", _, _] => in_vertex = false,
            ["property", ty, name] if in_vertex => {
                let scalar_type = ScalarType::parse(ty)
                    .with_context(|| format!("unsupported vertex property type `{ty}`"))?;
                vertex_properties.push(Property {
                    name: (*name).to_string(),
                    scalar_type,
                });
            }
            ["property", "list", ..] if in_vertex => {
                bail!("list properties on vertex elements are not supported");
            }
            _ => {}
        }
    }

    let format = format.context("PLY header is missing format line")?;
    let vertex_count = vertex_count.ok_or(LoadError::MissingVertexElement)?;
    validate_required_properties(&vertex_properties)?;

    Ok(PlyHeader {
        format,
        vertex_count,
        vertex_properties,
        data_start,
    })
}

fn validate_required_properties(properties: &[Property]) -> Result<()> {
    for name in [
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ] {
        if !properties.iter().any(|property| property.name == name) {
            return Err(LoadError::MissingProperty(name).into());
        }
    }
    Ok(())
}

fn parse_ascii_vertices(bytes: &[u8], header: &PlyHeader) -> Result<Vec<GaussianRaw>> {
    let reader = BufReader::new(bytes);
    let mut rows = Vec::with_capacity(header.vertex_count);

    for (line_index, line) in reader.lines().take(header.vertex_count).enumerate() {
        let line = line.context("failed to read ASCII PLY vertex line")?;
        let values: Vec<f32> = line
            .split_whitespace()
            .map(|value| value.parse::<f32>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .with_context(|| {
                format!("invalid float on ASCII PLY vertex line {}", line_index + 1)
            })?;
        if values.len() < header.vertex_properties.len() {
            bail!(
                "ASCII PLY vertex line {} has {} values, expected {}",
                line_index + 1,
                values.len(),
                header.vertex_properties.len()
            );
        }
        let map = header
            .vertex_properties
            .iter()
            .zip(values)
            .map(|(property, value)| (property.name.as_str(), value))
            .collect::<HashMap<_, _>>();
        rows.push(row_to_raw(&map, header));
    }

    Ok(rows)
}

fn parse_binary_vertices(bytes: &[u8], header: &PlyHeader) -> Result<Vec<GaussianRaw>> {
    let mut cursor = 0;
    let mut rows = Vec::with_capacity(header.vertex_count);

    for _ in 0..header.vertex_count {
        let mut map = HashMap::with_capacity(header.vertex_properties.len());
        for property in &header.vertex_properties {
            let value = property.scalar_type.read_f32(bytes, &mut cursor)?;
            map.insert(property.name.as_str(), value);
        }
        rows.push(row_to_raw(&map, header));
    }

    Ok(rows)
}

fn row_to_raw(values: &HashMap<&str, f32>, header: &PlyHeader) -> GaussianRaw {
    let mut f_rest_names: Vec<_> = header
        .vertex_properties
        .iter()
        .filter_map(|property| {
            property
                .name
                .strip_prefix("f_rest_")
                .and_then(|index| index.parse::<usize>().ok())
                .map(|index| (index, property.name.as_str()))
        })
        .collect();
    f_rest_names.sort_by_key(|(index, _)| *index);

    GaussianRaw {
        position: Vec3::new(values["x"], values["y"], values["z"]),
        f_dc: [values["f_dc_0"], values["f_dc_1"], values["f_dc_2"]],
        f_rest: f_rest_names
            .into_iter()
            .map(|(_, name)| values.get(name).copied().unwrap_or(0.0))
            .collect(),
        opacity_logit: values["opacity"],
        log_scale: Vec3::new(values["scale_0"], values["scale_1"], values["scale_2"]),
        rotation_raw: Vec4::new(
            values["rot_0"],
            values["rot_1"],
            values["rot_2"],
            values["rot_3"],
        ),
    }
}

fn read_array<const N: usize>(bytes: &[u8], cursor: &mut usize) -> Result<[u8; N]> {
    let end = *cursor + N;
    let slice = bytes
        .get(*cursor..end)
        .context("binary PLY ended unexpectedly while reading vertex data")?;
    *cursor = end;
    Ok(slice.try_into().expect("slice length is checked"))
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    #[test]
    fn ascii_loader_accepts_minimal_graphdeco_schema() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        write!(
            file,
            "ply\nformat ascii 1.0\nelement vertex 1\n\
             property float x\nproperty float y\nproperty float z\n\
             property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n\
             property float opacity\n\
             property float scale_0\nproperty float scale_1\nproperty float scale_2\n\
             property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n\
             end_header\n\
             1 2 3 0 0 0 0 -2 -2 -2 1 0 0 0\n"
        )
        .unwrap();
        let scene = load_scene(file.path(), LoadOptions::default()).unwrap();
        assert_eq!(scene.len(), 1);
        assert_eq!(scene.raw[0].position, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn binary_loader_accepts_minimal_graphdeco_schema() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(
            b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n\
              property float x\nproperty float y\nproperty float z\n\
              property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n\
              property float opacity\n\
              property float scale_0\nproperty float scale_1\nproperty float scale_2\n\
              property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n\
              end_header\n",
        )
        .unwrap();
        for value in [
            1.0_f32, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, -2.0, -2.0, -2.0, 1.0, 0.0, 0.0, 0.0,
        ] {
            file.write_all(&value.to_le_bytes()).unwrap();
        }

        let scene = load_scene(file.path(), LoadOptions::default()).unwrap();
        assert_eq!(scene.len(), 1);
        assert_eq!(scene.raw[0].position, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn loader_rejects_missing_required_property() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        write!(
            file,
            "ply\nformat ascii 1.0\nelement vertex 1\n\
             property float x\nproperty float y\nproperty float z\n\
             end_header\n0 0 0\n"
        )
        .unwrap();
        let error = load_scene(file.path(), LoadOptions::default())
            .unwrap_err()
            .to_string();
        assert!(error.contains("f_dc_0"));
    }

    #[test]
    fn max_splats_caps_loaded_rows() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        write!(
            file,
            "ply\nformat ascii 1.0\nelement vertex 2\n\
             property float x\nproperty float y\nproperty float z\n\
             property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n\
             property float opacity\n\
             property float scale_0\nproperty float scale_1\nproperty float scale_2\n\
             property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n\
             end_header\n\
             0 0 0 0 0 0 0 -2 -2 -2 1 0 0 0\n\
             1 1 1 0 0 0 0 -2 -2 -2 1 0 0 0\n"
        )
        .unwrap();
        let scene = load_scene(
            file.path(),
            LoadOptions {
                max_splats: Some(1),
                ..LoadOptions::default()
            },
        )
        .unwrap();
        assert_eq!(scene.len(), 1);
    }

    #[test]
    fn max_splats_prefers_high_importance_gaussians() {
        let raw = vec![
            sample_raw(Vec3::new(0.0, 0.0, 0.0), -8.0, Vec3::splat(-5.0)),
            sample_raw(Vec3::new(1.0, 0.0, 0.0), 8.0, Vec3::splat(-1.0)),
            sample_raw(Vec3::new(2.0, 0.0, 0.0), 0.0, Vec3::splat(-4.0)),
        ];

        let sampled = select_most_important(raw, 1);

        assert_eq!(sampled.len(), 1);
        assert_eq!(sampled[0].position, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn max_splats_keeps_original_order_after_importance_selection() {
        let raw = vec![
            sample_raw(Vec3::new(0.0, 0.0, 0.0), 8.0, Vec3::splat(-1.0)),
            sample_raw(Vec3::new(1.0, 0.0, 0.0), -8.0, Vec3::splat(-5.0)),
            sample_raw(Vec3::new(2.0, 0.0, 0.0), 7.0, Vec3::splat(-1.0)),
        ];

        let sampled = select_most_important(raw, 2);
        let positions = sampled
            .into_iter()
            .map(|item| item.position.x)
            .collect::<Vec<_>>();

        assert_eq!(positions, vec![0.0, 2.0]);
    }

    #[test]
    fn filter_splats_drops_low_opacity_gaussians() {
        let raw = vec![
            sample_raw(Vec3::ZERO, -8.0, Vec3::splat(-2.0)),
            sample_raw(Vec3::X, 8.0, Vec3::splat(-2.0)),
        ];

        let filtered = filter_splats(
            raw,
            LoadOptions {
                min_opacity: Some(0.5),
                ..LoadOptions::default()
            },
        );

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].position, Vec3::X);
    }

    #[test]
    fn filter_splats_drops_large_world_scale_gaussians() {
        let raw = vec![
            sample_raw(Vec3::ZERO, 8.0, Vec3::splat(0.0)),
            sample_raw(Vec3::X, 8.0, Vec3::splat(-2.0)),
        ];

        let filtered = filter_splats(
            raw,
            LoadOptions {
                max_world_scale: Some(0.5),
                ..LoadOptions::default()
            },
        );

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].position, Vec3::X);
    }

    fn sample_raw(position: Vec3, opacity_logit: f32, log_scale: Vec3) -> GaussianRaw {
        GaussianRaw {
            position,
            f_dc: [0.0, 0.0, 0.0],
            f_rest: Vec::new(),
            opacity_logit,
            log_scale,
            rotation_raw: Vec4::new(1.0, 0.0, 0.0, 0.0),
        }
    }
}
