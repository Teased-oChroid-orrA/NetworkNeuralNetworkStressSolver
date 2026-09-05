use std::{collections::HashMap, env, fs, path::Path};

use pinn_core::{
    messages::{ProblemKind, SolverConfig},
    units::{IN_TO_M, KSI_TO_PA, MSI_TO_PA},
    HoleType,
};

/// Read a KEY=VALUE env file; strip comments and blank lines.
fn load_pinn_env(path: &Path) -> HashMap<String, String> {
    let text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return HashMap::new(),
    };
    let mut map = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        // Strip inline comments before parsing
        let line = if let Some(pos) = line.find('#') { &line[..pos] } else { line };
        let line = line.trim();
        if line.is_empty() { continue; }
        if let Some((k, v)) = line.split_once('=') {
            let key = k.trim().to_string();
            let val = v.trim().to_string();
            if !key.is_empty() {
                map.insert(key, val);
            }
        }
    }
    map
}

/// Apply parsed env map to config in-place. Unknown keys are silently ignored.
///
/// `skip_problem_specific`: when `true`, the material/load/geometry overrides below are NOT
/// applied. `pinn.env`'s MATERIAL_E_MSI/MATERIAL_NU/LOAD_*/GEOM_*/HOLE_RADIUS_IN keys are
/// tuned for the Kirsch problem (e.g. Al 7075-T6, 10 ksi far-field tension) and would silently
/// overwrite `SolverConfig::default_pinlug()`'s fixed 4340-steel material with those Kirsch
/// defaults otherwise — pass `true` for `ProblemKind::PinLug`, whose material/geometry/load are
/// part of the problem definition, not user-tunable via this shared env file in this slice.
fn apply_env(cfg: &mut SolverConfig, env: &HashMap<String, String>, skip_problem_specific: bool) {
    macro_rules! parse_usize {
        ($key:expr, $field:expr) => {
            if let Some(v) = env.get($key) {
                if let Ok(n) = v.parse::<usize>() { $field = n; }
            }
        };
    }
    macro_rules! parse_f32 {
        ($key:expr, $field:expr) => {
            if let Some(v) = env.get($key) {
                if let Ok(n) = v.parse::<f32>() { $field = n; }
            }
        };
    }
    macro_rules! parse_f64 {
        ($key:expr, $field:expr) => {
            if let Some(v) = env.get($key) {
                if let Ok(n) = v.parse::<f64>() { $field = n; }
            }
        };
    }
    macro_rules! parse_bool {
        ($key:expr, $field:expr) => {
            if let Some(v) = env.get($key) {
                match v.to_lowercase().as_str() {
                    "true" | "1" | "yes" => $field = true,
                    "false" | "0" | "no" => $field = false,
                    _ => {}
                }
            }
        };
    }

    // Sampling & schedule
    parse_usize!("N_INTERIOR", cfg.n_interior);
    parse_usize!("N_BOUNDARY", cfg.n_boundary);
    parse_usize!("MAX_STEPS",  cfg.max_steps);
    if let Some(v) = env.get("VIS_GRID_NX") {
        if let Ok(n) = v.parse::<usize>() { cfg.vis_grid[0] = n; }
    }
    if let Some(v) = env.get("VIS_GRID_NY") {
        if let Ok(n) = v.parse::<usize>() { cfg.vis_grid[1] = n; }
    }

    // Network
    parse_usize!("HIDDEN_DIM", cfg.hidden_dim);
    parse_usize!("N_HIDDEN",   cfg.n_hidden);
    parse_f32!("FD_H",         cfg.fd_h);

    // Optimizer
    parse_bool!("USE_SOAP_MUON", cfg.use_soap_muon);
    parse_bool!("USE_PIRATENET", cfg.use_piratenet);
    parse_bool!("USE_PIRATENET_COMPUTE_SKIP", cfg.use_piratenet_compute_skip);

    // Decision maker
    {
        let dm = &mut cfg.decision_maker;
        parse_bool!("DM_ENABLED",               dm.enabled);
        parse_usize!("DM_CHECK_INTERVAL",        dm.check_interval);
        parse_f32!("DM_CONFLICT_THRESHOLD",      dm.conflict_threshold);
        parse_f32!("DM_ALIGNMENT_THRESHOLD",     dm.alignment_threshold);
        parse_f32!("DM_CONVERGE_COSINE_MIN",     dm.converge_cosine_min);
        parse_f32!("DM_CONVERGE_GRAD_THRESHOLD", dm.converge_grad_threshold);
        parse_bool!("DM_USE_EXACT_COSINE",       dm.use_exact_cosine);
        parse_usize!("DM_MIN_DWELL_STEPS",       dm.min_dwell_steps);
        parse_usize!("DM_LBFGS_MAX_ITER",        dm.lbfgs_max_iter);
    }

    // Stiffness-coupled SAW-BRDR / PirateNet-gate accelerator
    {
        let st = &mut cfg.stiffness;
        parse_bool!("STIFF_ENABLED",             st.enabled);
        parse_usize!("STIFF_CHECK_INTERVAL",     st.check_interval);
        parse_f32!("STIFF_EMA_BETA",             st.ema_beta);
        parse_f32!("STIFF_PHYSICS_BOOST_GAIN",   st.physics_boost_gain);
        parse_f32!("STIFF_ALPHA_ACCEL_GAIN",     st.alpha_accel_gain);
        parse_f32!("STIFF_GATE_AWAKE_EPSILON",   st.gate_awake_epsilon);
    }

    // Execution / performance profile (hardware-adaptive-execution epic, Phase 1). Neither
    // key changes any executed code path yet - see `pinn_solver::execution`'s module doc for
    // why (profiling-driven optimization, not a guess, decides what each value should
    // concretely do). No existing macro parses enums/strings, so this is a plain match block
    // rather than a new one-off macro for just two keys, mirroring `parse_problem_arg`'s own
    // string-match style below.
    {
        let ex = &mut cfg.execution;
        if let Some(v) = env.get("EXEC_MODE") {
            match v.to_lowercase().as_str() {
                "auto"   => ex.mode = pinn_core::messages::ExecutionMode::Auto,
                "serial" => ex.mode = pinn_core::messages::ExecutionMode::Serial,
                _ => {}
            }
        }
        if let Some(v) = env.get("EXEC_PROFILE") {
            match v.to_lowercase().as_str() {
                "eco"         => ex.profile = pinn_core::messages::PerformanceProfile::Eco,
                "balanced"    => ex.profile = pinn_core::messages::PerformanceProfile::Balanced,
                "performance" => ex.profile = pinn_core::messages::PerformanceProfile::Performance,
                "maximum"     => ex.profile = pinn_core::messages::PerformanceProfile::Maximum,
                _ => {}
            }
        }
    }

    // Per-step profiling instrumentation (hardware-adaptive-execution epic, Phase 2). Real,
    // opt-in wall-time measurement - see pinn_solver::diagnostics's module doc for the
    // device-sync cost this adds when enabled.
    parse_bool!("DIAGNOSTICS_ENABLED", cfg.diagnostics.enabled);

    if skip_problem_specific {
        return;
    }

    // Material (US Customary → SI)
    if let Some(v) = env.get("MATERIAL_E_MSI") {
        if let Ok(n) = v.parse::<f64>() { cfg.material.e = n * MSI_TO_PA; }
    }
    parse_f64!("MATERIAL_NU", cfg.material.nu);

    // Load (ksi → Pa)
    if let Some(v) = env.get("LOAD_PX_KSI") {
        if let Ok(n) = v.parse::<f64>() { cfg.load.px = n * KSI_TO_PA; }
    }
    if let Some(v) = env.get("LOAD_PY_KSI") {
        if let Ok(n) = v.parse::<f64>() { cfg.load.py = n * KSI_TO_PA; }
    }

    // Geometry (inches → m)
    if let Some(v) = env.get("GEOM_HALF_W_IN") {
        if let Ok(n) = v.parse::<f64>() { cfg.geometry.half_w = n * IN_TO_M; }
    }
    if let Some(v) = env.get("GEOM_HALF_H_IN") {
        if let Ok(n) = v.parse::<f64>() { cfg.geometry.half_h = n * IN_TO_M; }
    }
    if let Some(v) = env.get("HOLE_RADIUS_IN") {
        if let Ok(n) = v.parse::<f64>() {
            cfg.geometry.hole = HoleType::Circular { radius: n * IN_TO_M };
        }
    }
}

/// Parse `--problem kirsch|pinlug` from argv (defaults to `kirsch` — the existing,
/// unaffected behavior — if the flag is absent or has an unrecognized value).
fn parse_problem_arg() -> ProblemKind {
    let args: Vec<String> = env::args().collect();
    for i in 0..args.len() {
        if args[i] == "--problem" {
            if let Some(v) = args.get(i + 1) {
                return match v.as_str() {
                    "pinlug" => ProblemKind::PinLug,
                    _ => ProblemKind::Kirsch,
                };
            }
        }
    }
    ProblemKind::Kirsch
}

fn main() -> anyhow::Result<()> {
    let env_path_str = env::var("PINN_ENV").unwrap_or_else(|_| "pinn.env".to_string());
    let env_map = load_pinn_env(Path::new(&env_path_str));

    let problem_kind = parse_problem_arg();
    let mut config = match problem_kind {
        ProblemKind::Kirsch => SolverConfig::default_kirsch(),
        ProblemKind::PinLug => SolverConfig::default_pinlug(),
    };
    apply_env(&mut config, &env_map, problem_kind == ProblemKind::PinLug);

    if !env_map.is_empty() {
        eprintln!("[pinn.env] loaded {} key(s) from {env_path_str}",  env_map.len());
    }

    let headless = env::args().any(|a| a == "--headless" || a == "-H");

    if headless {
        let ok = match problem_kind {
            // Routes through the step_physics_multi 2-domain driver, not run_headless's
            // frozen 1-domain Kirsch step_physics path.
            ProblemKind::PinLug => pinn_solver::run_headless_pinlug(config),
            ProblemKind::Kirsch => pinn_solver::run_headless(config),
        };
        if !ok {
            std::process::exit(1);
        }
        return Ok(());
    }

    // GUI mode
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 820.0])
            .with_min_inner_size([900.0, 600.0])
            .with_title("PINN Structural Stress Solver"),
        ..Default::default()
    };

    eframe::run_native(
        "PINN Stress Solver",
        native_options,
        Box::new(move |cc| Ok(Box::new(pinn_gui::StressSolverApp::new(cc, config, problem_kind)))),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_problem_arg_default_matches_pinn_core_kirsch() {
        let k: pinn_core::messages::ProblemKind = pinn_core::messages::ProblemKind::Kirsch;
        assert_eq!(k, pinn_core::messages::ProblemKind::Kirsch);
    }

    #[test]
    fn apply_env_parses_exec_mode_and_exec_profile() {
        let mut cfg = SolverConfig::default_kirsch();
        let mut env = HashMap::new();
        env.insert("EXEC_MODE".to_string(), "serial".to_string());
        env.insert("EXEC_PROFILE".to_string(), "performance".to_string());
        apply_env(&mut cfg, &env, false);
        assert_eq!(cfg.execution.mode, pinn_core::messages::ExecutionMode::Serial);
        assert_eq!(cfg.execution.profile, pinn_core::messages::PerformanceProfile::Performance);
    }

    #[test]
    fn apply_env_ignores_unrecognized_exec_mode_and_exec_profile_values() {
        let mut cfg = SolverConfig::default_kirsch();
        let mut env = HashMap::new();
        env.insert("EXEC_MODE".to_string(), "quantum".to_string());
        env.insert("EXEC_PROFILE".to_string(), "ludicrous".to_string());
        apply_env(&mut cfg, &env, false);
        // Unrecognized values leave the (default) config untouched, matching this file's
        // existing "unknown keys are silently ignored" convention.
        assert_eq!(cfg.execution.mode, pinn_core::messages::ExecutionMode::Auto);
        assert_eq!(cfg.execution.profile, pinn_core::messages::PerformanceProfile::Balanced);
    }

    #[test]
    fn apply_env_exec_keys_are_not_skipped_by_skip_problem_specific() {
        // Execution config is not part of the problem definition (unlike material/load/
        // geometry) - it must still apply when skip_problem_specific=true (the pin-lug path).
        let mut cfg = SolverConfig::default_pinlug();
        let mut env = HashMap::new();
        env.insert("EXEC_MODE".to_string(), "serial".to_string());
        apply_env(&mut cfg, &env, true);
        assert_eq!(cfg.execution.mode, pinn_core::messages::ExecutionMode::Serial);
    }

    #[test]
    fn apply_env_parses_diagnostics_enabled() {
        let mut cfg = SolverConfig::default_kirsch();
        assert!(!cfg.diagnostics.enabled, "must default to disabled");
        let mut env = HashMap::new();
        env.insert("DIAGNOSTICS_ENABLED".to_string(), "true".to_string());
        apply_env(&mut cfg, &env, false);
        assert!(cfg.diagnostics.enabled);
    }
}
