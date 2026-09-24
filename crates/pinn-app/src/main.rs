use std::{collections::HashMap, env, fs, path::Path};

use pinn_core::{
    messages::{ProblemKind, SolverConfig},
    units::{IN_TO_M, KSI_TO_PA, MSI_TO_PA},
    HoleType,
};

mod tui;

/// Read a KEY=VALUE env file; strip comments and blank lines.
fn load_pinn_env(path: &Path) -> anyhow::Result<HashMap<String, String>> {
    let text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && env::var_os("PINN_ENV").is_none() => return Ok(HashMap::new()),
        Err(e) => anyhow::bail!("cannot read configuration {}: {e}", path.display()),
    };
    let mut map = HashMap::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        // Strip inline comments before parsing
        let line = if let Some(pos) = line.find('#') { &line[..pos] } else { line };
        let line = line.trim();
        if line.is_empty() { continue; }
        if let Some((k, v)) = line.split_once('=') {
            let key = k.trim().to_string();
            let val = v.trim().to_string();
            anyhow::ensure!(!key.is_empty() && !val.is_empty(), "{}:{}: expected KEY=VALUE", path.display(), index + 1);
            anyhow::ensure!(!map.contains_key(&key), "{}:{}: duplicate key {key}", path.display(), index + 1);
            map.insert(key, val);
        } else {
            anyhow::bail!("{}:{}: expected KEY=VALUE", path.display(), index + 1);
        }
    }
    Ok(map)
}

/// Apply a validated env map. Invalid and unknown values are configuration errors.
///
/// `skip_problem_specific`: when `true`, the material/load/geometry overrides below are NOT
/// applied. `pinn.env`'s MATERIAL_E_MSI/MATERIAL_NU/LOAD_*/GEOM_*/HOLE_RADIUS_IN keys are
/// tuned for the Kirsch problem (e.g. Al 7075-T6, 10 ksi far-field tension) and would silently
/// overwrite `SolverConfig::default_pinlug()`'s fixed 4340-steel material with those Kirsch
/// defaults otherwise — pass `true` for `ProblemKind::PinLug`, whose material/geometry/load are
/// part of the problem definition, not user-tunable via this shared env file in this slice.
fn apply_env(cfg: &mut SolverConfig, env: &HashMap<String, String>, skip_problem_specific: bool) -> anyhow::Result<()> {
    validate_env(env)?;
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
        return Ok(());
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
    anyhow::ensure!(cfg.geometry.half_w.is_finite() && cfg.geometry.half_h.is_finite() && cfg.material.e.is_finite() && cfg.load.px.is_finite() && cfg.load.py.is_finite(), "physical values overflow after SI conversion");
    if let HoleType::Circular { radius } = cfg.geometry.hole {
        anyhow::ensure!(radius.is_finite() && radius < cfg.geometry.half_w.min(cfg.geometry.half_h), "hole radius must be smaller than both plate half-dimensions");
    }
    Ok(())
}

/// Validate before applying values so a malformed override never silently keeps a default.
fn validate_env(values: &HashMap<String, String>) -> anyhow::Result<()> {
    for (key, value) in values {
        let valid = match key.as_str() {
            "N_INTERIOR" | "N_BOUNDARY" | "MAX_STEPS" | "HIDDEN_DIM" | "N_HIDDEN"
            | "DM_CHECK_INTERVAL" | "DM_MIN_DWELL_STEPS" | "DM_LBFGS_MAX_ITER"
            | "STIFF_CHECK_INTERVAL" => value.parse::<usize>().is_ok_and(|n| n > 0),
            "VIS_GRID_NX" | "VIS_GRID_NY" => value.parse::<usize>().is_ok_and(|n| n >= 2),
            "USE_SOAP_MUON" | "USE_PIRATENET" | "USE_PIRATENET_COMPUTE_SKIP"
            | "DM_ENABLED" | "DM_USE_EXACT_COSINE" | "STIFF_ENABLED" | "DIAGNOSTICS_ENABLED"
                => matches!(value.to_ascii_lowercase().as_str(), "true" | "false" | "1" | "0" | "yes" | "no"),
            "EXEC_MODE" => matches!(value.to_ascii_lowercase().as_str(), "auto" | "serial"),
            "EXEC_PROFILE" => matches!(value.to_ascii_lowercase().as_str(), "eco" | "balanced" | "performance" | "maximum"),
            "MATERIAL_NU" => value.parse::<f64>().is_ok_and(|n| n.is_finite() && n > -1.0 && n < 0.5),
            "LOAD_PX_KSI" | "LOAD_PY_KSI" => value.parse::<f64>().is_ok_and(|n| n.is_finite() && (n * KSI_TO_PA).is_finite()),
            "MATERIAL_E_MSI" | "GEOM_HALF_W_IN" | "GEOM_HALF_H_IN" | "HOLE_RADIUS_IN"
                => value.parse::<f64>().is_ok_and(|n| n.is_finite() && n > 0.0),
            "DM_CONFLICT_THRESHOLD" | "DM_ALIGNMENT_THRESHOLD" | "DM_CONVERGE_COSINE_MIN"
                => value.parse::<f32>().is_ok_and(|n| n.is_finite() && (-1.0..=1.0).contains(&n)),
            "STIFF_EMA_BETA" => value.parse::<f32>().is_ok_and(|n| n.is_finite() && (0.0..1.0).contains(&n)),
            "FD_H" => value.parse::<f32>().is_ok_and(|n| n.is_finite() && n > 0.0 && n < 1.0),
            "DM_CONVERGE_GRAD_THRESHOLD" | "STIFF_GATE_AWAKE_EPSILON"
            | "STIFF_PHYSICS_BOOST_GAIN" | "STIFF_ALPHA_ACCEL_GAIN"
                => value.parse::<f32>().is_ok_and(|n| n.is_finite() && n >= 0.0),
            _ => anyhow::bail!("unknown configuration key '{key}'"),
        };
        anyhow::ensure!(valid, "invalid configuration value for '{key}': '{value}'");
    }
    Ok(())
}

#[derive(Default)]
struct Arguments {
    problem: Option<ProblemKind>,
    spec: Option<String>,
    tui: bool,
    headless: bool,
    help: bool,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> anyhow::Result<Arguments> {
    let mut parsed = Arguments::default();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tui" | "-T" => parsed.tui = true,
            "--headless" | "-H" => parsed.headless = true,
            "--help" | "-h" => parsed.help = true,
            "--problem" | "--problem-spec" => {
                let value = args.next().filter(|s| !s.starts_with('-'))
                    .ok_or_else(|| anyhow::anyhow!("{arg} requires a value"))?;
                if arg == "--problem" {
                    anyhow::ensure!(parsed.problem.is_none(), "duplicate --problem");
                    parsed.problem = Some(match value.as_str() {
                        "kirsch" => ProblemKind::Kirsch,
                        "pinlug" => ProblemKind::PinLug,
                        _ => anyhow::bail!("unknown problem '{value}'; expected kirsch or pinlug"),
                    });
                } else {
                    anyhow::ensure!(parsed.spec.is_none(), "duplicate --problem-spec");
                    parsed.spec = Some(value);
                }
            }
            _ => anyhow::bail!("unknown argument '{arg}'"),
        }
    }
    anyhow::ensure!(parsed.problem.is_none() || parsed.spec.is_none(), "--problem and --problem-spec are mutually exclusive");
    Ok(parsed)
}

fn main() -> anyhow::Result<()> {
    let args = parse_args(env::args().skip(1))?;
    if args.help {
        println!("stress-solver [--tui|-T | --headless|-H] [--problem kirsch|pinlug | --problem-spec PATH]\nPINN_ENV selects a strict KEY=VALUE configuration file for built-in problems.");
        return Ok(());
    }
    // `--tui`/`-T`: a terminal dashboard, reached and dispatched FIRST, before either the
    // `--problem-spec` early return or `--headless` below — both being other "no window" run
    // modes, `--tui` wins if given alongside them (the strictly more capable request). Every
    // path this check falls through to (`--headless`, `--problem-spec` without `--tui`, plain
    // GUI) is completely unaffected — this function returns before any of that code runs.
    let tui = args.tui;
    if tui {
        if let Some(spec_path) = args.spec.as_ref() {
            let spec_str = std::fs::read_to_string(&spec_path)
                .map_err(|e| anyhow::anyhow!("failed to read --problem-spec file '{spec_path}': {e}"))?;
            let spec: pinn_core::problem_spec::ProblemSpec = toml::from_str(&spec_str)
                .map_err(|e| anyhow::anyhow!("failed to parse --problem-spec TOML '{spec_path}': {e}"))?;
            spec.geometry.validate().map_err(|e| anyhow::anyhow!("invalid geometry in '{spec_path}': {e}"))?;
            return tui::run_tui_plate(spec);
        }
        let problem_explicit = args.problem.is_some();
        let problem_kind = if problem_explicit {
            args.problem
        } else {
            // Neither `--problem` nor `--problem-spec` was given - ask interactively instead of
            // silently defaulting to Kirsch, so `--tui` alone is a genuinely self-contained
            // entry point (problem selection AND, for User-Defined, spec loading both happen
            // inside the TUI itself).
            match tui::run_setup_menu()? {
                None => return Ok(()),
                Some(tui::SetupChoice::Kirsch) => Some(ProblemKind::Kirsch),
                Some(tui::SetupChoice::PinLug) => Some(ProblemKind::PinLug),
                Some(tui::SetupChoice::Plate(spec)) => return tui::run_tui_plate(spec),
            }
        };
        let problem_kind = problem_kind.expect("set to Some on every non-early-return path above");
        let env_path_str = env::var("PINN_ENV").unwrap_or_else(|_| "pinn.env".to_string());
        let env_map = load_pinn_env(Path::new(&env_path_str))?;
        let mut config = match problem_kind {
            ProblemKind::Kirsch => SolverConfig::default_kirsch(),
            ProblemKind::PinLug => SolverConfig::default_pinlug(),
        };
        apply_env(&mut config, &env_map, problem_kind == ProblemKind::PinLug)?;
        return tui::run_tui_live(config, problem_kind);
    }

    // User-defined-problem ingestion: checked BEFORE any Kirsch/pin-lug dispatch below, so
    // that dispatch (and pinn.env loading, which is irrelevant to a self-contained spec
    // file) is completely untouched when this flag is absent — the "default to the
    // already-defined hardcoded problems" behavior this feature was required to preserve.
    if let Some(spec_path) = args.spec.as_ref() {
        let spec_str = std::fs::read_to_string(&spec_path)
            .map_err(|e| anyhow::anyhow!("failed to read --problem-spec file '{spec_path}': {e}"))?;
        let spec: pinn_core::problem_spec::ProblemSpec = toml::from_str(&spec_str)
            .map_err(|e| anyhow::anyhow!("failed to parse --problem-spec TOML '{spec_path}': {e}"))?;
        // Issue #78 Stage 1.2: fail fast on a geometrically-nonsensical spec (a hole outside
        // the plate, two holes overlapping) before spending any time constructing a problem or
        // starting training on it.
        spec.geometry.validate().map_err(|e| anyhow::anyhow!("invalid geometry in '{spec_path}': {e}"))?;
        let ok = pinn_solver::user_runner::run_headless_user_problem(spec);
        return if ok {
            Ok(())
        } else {
            anyhow::bail!("user-defined problem training ended with a non-finite loss")
        };
    }

    let env_path_str = env::var("PINN_ENV").unwrap_or_else(|_| "pinn.env".to_string());
    let env_map = load_pinn_env(Path::new(&env_path_str))?;

    let problem_kind = args.problem.unwrap_or(ProblemKind::Kirsch);
    let mut config = match problem_kind {
        ProblemKind::Kirsch => SolverConfig::default_kirsch(),
        ProblemKind::PinLug => SolverConfig::default_pinlug(),
    };
    apply_env(&mut config, &env_map, problem_kind == ProblemKind::PinLug)?;

    // Hardware-adaptive-execution epic, Phase 4. `apply_performance_profile` (the Eco
    // n_interior/n_boundary reduction) is NOT called here - Kirsch's `run_headless_inner` runs
    // `EngineParams::apply_to` first, which unconditionally overwrites those fields from
    // geometry analysis, silently clobbering any reduction applied this early. Each headless
    // entry point (`run_headless_inner`, `run_headless_pinlug_inner`) calls it itself at the
    // correct point instead - see those call sites' own comments.
    if let Some(threads) = pinn_solver::execution::cpu_thread_count(config.execution.profile) {
        // Configures rayon's PROCESS-GLOBAL thread pool once, before any tensor work starts -
        // most relevant to burn-ndarray's own internal rayon usage (--features
        // ndarray-backend), a harmless setting otherwise (default Wgpu backend's tensor
        // compute is GPU-bound). Errors (e.g. a second call attempting to rebuild an
        // already-initialized global pool) are intentionally ignored - this is a best-effort
        // resource cap, never worth failing startup over.
        let _ = rayon::ThreadPoolBuilder::new().num_threads(threads).build_global();
    }

    if !env_map.is_empty() {
        eprintln!("[pinn.env] loaded {} key(s) from {env_path_str}",  env_map.len());
    }

    let headless = args.headless;

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
        apply_env(&mut cfg, &env, false).unwrap();
        assert_eq!(cfg.execution.mode, pinn_core::messages::ExecutionMode::Serial);
        assert_eq!(cfg.execution.profile, pinn_core::messages::PerformanceProfile::Performance);
    }

    #[test]
    fn apply_env_rejects_unrecognized_exec_mode_and_exec_profile_values() {
        let mut cfg = SolverConfig::default_kirsch();
        let mut env = HashMap::new();
        env.insert("EXEC_MODE".to_string(), "quantum".to_string());
        env.insert("EXEC_PROFILE".to_string(), "ludicrous".to_string());
        assert!(apply_env(&mut cfg, &env, false).is_err());
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
        apply_env(&mut cfg, &env, true).unwrap();
        assert_eq!(cfg.execution.mode, pinn_core::messages::ExecutionMode::Serial);
    }

    #[test]
    fn apply_env_parses_diagnostics_enabled() {
        let mut cfg = SolverConfig::default_kirsch();
        assert!(!cfg.diagnostics.enabled, "must default to disabled");
        let mut env = HashMap::new();
        env.insert("DIAGNOSTICS_ENABLED".to_string(), "true".to_string());
        apply_env(&mut cfg, &env, false).unwrap();
        assert!(cfg.diagnostics.enabled);
    }
    #[test]
    fn malformed_configuration_is_rejected_before_mutation() {
        for (key, value) in [("MATERIAL_E_MSI", "abc"), ("FD_H", "NaN"),
            ("MAX_STEPS", "0"), ("MATERIAL_NU", "0.5"), ("DM_ENABLED", "maybe"),
            ("LOAD_PX_KSI", "inf"), ("MAX_STEP", "10")] {
            let mut cfg = SolverConfig::default_kirsch();
            let original = cfg.max_steps;
            let values = HashMap::from([(key.to_owned(), value.to_owned())]);
            assert!(apply_env(&mut cfg, &values, false).is_err(), "{key}");
            assert_eq!(cfg.max_steps, original);
        }
    }

    #[test]
    fn cli_rejects_typos_missing_values_and_conflicting_problem_sources() {
        for args in [vec!["--problem", "pinlgu"], vec!["--problem-spec"],
            vec!["--problem", "--headless"], vec!["--unknown"],
            vec!["--problem", "kirsch", "--problem-spec", "p.toml"]] {
            assert!(parse_args(args.into_iter().map(str::to_owned)).is_err());
        }
        let parsed = parse_args(["--headless", "--problem", "pinlug"].map(str::to_owned)).unwrap();
        assert!(parsed.headless);
        assert_eq!(parsed.problem, Some(ProblemKind::PinLug));
        assert!(parse_args(Vec::<String>::new()).unwrap().problem.is_none());
    }

}
