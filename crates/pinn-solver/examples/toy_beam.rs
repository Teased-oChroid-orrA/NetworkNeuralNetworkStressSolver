//! Runnable driver for the standalone 1D beam sanity check (see
//! `pinn_solver::toy_beam`'s module doc for the why/how). Trains both boundary-condition
//! cases and prints a side-by-side network-vs-exact comparison, so the training
//! methodology (does it actually minimize the residual, or collapse to a trivial
//! near-zero deflection?) can be checked in seconds instead of the ~70-90 minutes the
//! real Kirsch/pin-lug default config takes.
//!
//! Run: `cargo run -p pinn-solver --example toy_beam --release`

use pinn_solver::toy_beam::{train_toy_beam, BeamBc};

fn run_case(name: &str, bc: BeamBc) {
    println!("=== {name} ===");
    let result = train_toy_beam(bc, 3000, 32, 48, 3);

    println!("{:>6}  {:>12}  {:>12}  {:>12}", "x", "w_net", "w_exact", "abs_err");
    for (x, w_net, w_exact) in &result.eval_points {
        println!("{x:>6.2}  {w_net:>12.6}  {w_exact:>12.6}  {:>12.6}", (w_net - w_exact).abs());
    }

    println!(
        "final_loss={:.6e}  max_abs_error={:.6e}  max_abs_deflection={:.6e}",
        result.final_loss, result.max_abs_error, result.max_abs_deflection
    );
    if result.max_abs_deflection < 1e-4 {
        println!("  [!] max_abs_deflection is suspiciously small — possible trivial-solution collapse.");
    } else {
        println!("  [ok] deflection is non-trivial.");
    }
    println!();
}

fn main() {
    run_case("Cantilever (clamped-free)", BeamBc::Cantilever);
    run_case("Simply supported (pinned-pinned)", BeamBc::SimplySupported);
}
