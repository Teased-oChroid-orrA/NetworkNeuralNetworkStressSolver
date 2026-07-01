use egui::Color32;
use ndarray::Array2;

/// 8-breakpoint piecewise-linear Viridis colormap approximation
const VIRIDIS: &[(f32, u8, u8, u8)] = &[
    (0.000,  68,  1,  84),
    (0.143,  72, 40, 120),
    (0.286,  62, 83, 160),
    (0.429,  49,104, 142),
    (0.571,  53,183, 121),
    (0.714, 109,205,  89),
    (0.857, 180,222,  44),
    (1.000, 253,231,  37),
];

/// Map a normalised value t ∈ [0,1] → Viridis Color32
pub fn viridis(t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let i = VIRIDIS
        .partition_point(|&(x, ..)| x <= t)
        .saturating_sub(1)
        .min(VIRIDIS.len() - 2);
    let (t0, r0, g0, b0) = VIRIDIS[i];
    let (t1, r1, g1, b1) = VIRIDIS[i + 1];
    let s = if (t1 - t0).abs() < 1e-6 { 0.0 } else { (t - t0) / (t1 - t0) };
    Color32::from_rgb(
        lerp(r0, r1, s),
        lerp(g0, g1, s),
        lerp(b0, b1, s),
    )
}

fn lerp(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + t * (b as f32 - a as f32)) as u8
}

/// Convert a 2-D scalar field to a flat pixel buffer (row-major).
/// NaN values are rendered as dark grey.
///
/// `flip_rows`: read row `ny-1-i` instead of row `i` for output row `i` — used to put
/// physical y=0 at the bottom of the rendered image without first cloning and physically
/// swapping the whole array (the field is touched once, not twice).
pub fn field_to_pixels(field: &Array2<f32>, flip_rows: bool) -> (Vec<Color32>, f32, f32) {
    // Compute min/max ignoring NaN
    let mut vmin = f32::INFINITY;
    let mut vmax = f32::NEG_INFINITY;
    for &v in field.iter() {
        if v.is_finite() {
            vmin = vmin.min(v);
            vmax = vmax.max(v);
        }
    }
    if !vmin.is_finite() { vmin = 0.0; }
    if !vmax.is_finite() { vmax = 1.0; }
    let range = (vmax - vmin).max(1e-10);

    let (ny, nx) = field.dim();
    let mut pixels = Vec::with_capacity(nx * ny);
    for i in 0..ny {
        let src_row = if flip_rows { ny - 1 - i } else { i };
        for j in 0..nx {
            let v = field[[src_row, j]];
            pixels.push(if v.is_nan() {
                Color32::from_gray(30)
            } else {
                viridis((v - vmin) / range)
            });
        }
    }

    (pixels, vmin, vmax)
}
