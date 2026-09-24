//! Reproduce the GUI/TUI plate runner without rendering; persist authoritative telemetry.
//! Usage: cargo run -p pinn-solver --release --features ndarray-backend --example audit_trace -- SPEC NEW_OUTPUT_DIR [STEP_LIMIT]
use pinn_core::{
    messages::{ControlMsg, TrainingMsg},
    problem_spec::ProblemSpec,
};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
};
fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    anyhow::ensure!(
        (2..=3).contains(&args.len()),
        "usage: audit_trace SPEC NEW_OUTPUT_DIR [STEP_LIMIT]"
    );
    let original = fs::read_to_string(&args[0])?;
    let mut spec: ProblemSpec = toml::from_str(&original)?;
    spec.geometry.validate().map_err(anyhow::Error::msg)?;
    let requested_steps = spec.training.max_steps;
    if let Some(limit) = args.get(2) {
        let limit: usize = limit.parse()?;
        anyhow::ensure!(
            limit > 0 && limit <= requested_steps,
            "step limit must be in 1..=configured max_steps"
        );
        spec.training.max_steps = limit;
    }
    let dir = PathBuf::from(&args[1]);
    fs::create_dir(&dir)?;
    fs::write(dir.join("original.toml"), original)?;
    fs::write(dir.join("effective.toml"), toml::to_string_pretty(&spec)?)?;
    let git = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
    };
    fs::write(
        dir.join("manifest.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "source_spec": args[0], "requested_steps": requested_steps, "effective_steps": spec.training.max_steps,
            "purpose": "GUI/TUI shared-runner diagnostic; bounded runs are not convergence acceptance",
            "git_sha": git(&["rev-parse", "HEAD"]), "git_status": git(&["status", "--porcelain"]),
            "backend": if cfg!(feature="ndarray-backend") { "NdArray" } else { "WGPU" },
            "dtype": "f32", "resume": false, "seed": spec.network.model_init_seed,
        }))?,
    )?;
    let (tx, rx) = crossbeam_channel::bounded(64);
    let (control, controls) = crossbeam_channel::bounded(8);
    let worker = std::thread::spawn(move || {
        pinn_solver::runner::run_training_user_problem(spec, tx, controls)
    });
    let mut stream = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(dir.join("updates.jsonl"))?;
    let mut count = 0;
    let mut negative_total = 0;
    let mut negative_energy = 0;
    let mut negative_other = 0;
    let mut max_grad = 0.0_f32;
    let mut completed = false;
    let mut nonfinite = false;
    while let Ok(message) = rx.recv() {
        match message {
            TrainingMsg::Update(u) => {
                count += 1;
                negative_total += usize::from(u.total_loss < 0.0);
                negative_energy += usize::from(u.energy_loss < 0.0);
                negative_other += usize::from(u.neumann_loss < 0.0);
                max_grad = max_grad.max(u.grad_norm.unwrap_or(0.0));
                nonfinite |= !u.total_loss.is_finite()
                    || !u.energy_loss.is_finite()
                    || !u.neumann_loss.is_finite()
                    || u.grad_norm.is_some_and(|g| !g.is_finite());
                let row = serde_json::json!({ "step":u.step,"total_loss":u.total_loss,"energy_loss":u.energy_loss,
                    "legacy_neumann_channel":u.neumann_loss,"grad_norm":u.grad_norm,"lr":u.lr,"n_colloc":u.n_colloc,
                    "objective":u.objective,"energy_balance":u.energy_balance,"reaction_force":u.reaction_force,
                    "bc_rms":if u.vis.is_some(){Some(u.bc_residual_rms)}else{None},
                    "gradient_shares":u.gradient_share_report.as_ref().map(|g| &g.shares),
                    "gradient_conflicts":u.gradient_conflict_report.as_ref().map(|g| &g.pairs),
                    "holes":u.hole_analyses.iter().map(|h| serde_json::json!({"index":h.hole_index,"kt":h.concentration.kt})).collect::<Vec<_>>() });
                serde_json::to_writer(&mut stream, &row)?;
                writeln!(stream)?;
            }
            TrainingMsg::Done => {
                completed = true;
                break;
            }
            TrainingMsg::Error(e) => {
                writeln!(stream, "{}", serde_json::json!({"error":e}))?;
                break;
            }
            _ => {}
        }
    }
    let _ = control.try_send(ControlMsg::Stop);
    drop(rx);
    let worker_ok = worker.join().is_ok();
    stream.flush()?;
    let summary = serde_json::json!({"updates":count,"completed":completed,"worker_ok":worker_ok,
        "nonfinite":nonfinite,"negative_total_count":negative_total,"negative_energy_count":negative_energy,
        "negative_legacy_neumann_count":negative_other,"max_grad_norm":max_grad});
    fs::write(
        dir.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    println!("{summary}");
    anyhow::ensure!(
        completed && worker_ok && !nonfinite && count > 0,
        "diagnostic run failed; inspect persisted evidence"
    );
    Ok(())
}
