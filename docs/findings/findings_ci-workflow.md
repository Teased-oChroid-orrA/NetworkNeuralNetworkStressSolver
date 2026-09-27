# CI concurrency and push discipline

> Relocated from `CLAUDE.md` (this project's former single-file findings log) into its own topic file, per the project's "separate findings from CLAUDE.md" convention. The GitHub Actions `concurrency` cancel-in-progress group and the working-session convention of pushing to a feature branch, not `main`, per commit.

## CI concurrency + push discipline (GitHub Actions run-count control)

`.github/workflows/rust.yml` gained a `concurrency: group: ${{ github.workflow }}-${{ github.ref
}}, cancel-in-progress: true` block - a newer push to `main` cancels a still-running older CI run
on the same ref instead of letting both run to completion (~55-65 min each), since several
pushes to `main` in quick succession would otherwise queue up and burn that wall-clock N times
over for only the latest commit's result mattering. The real lever, though, is push discipline:
commit work to the feature branch (`fix/issue-77-kt-root-cause`) as it lands - pushing a
non-`main` branch triggers no workflow run at all, since `rust.yml`'s `on.push.branches` is
`["main"]` only - and fast-forward + push `main` once, after everything in a batch of work is
verified, rather than once per commit. This is a working-session convention, not a repository
rule enforced anywhere in code - worth restating explicitly if a future session's own pattern
drifts back toward pushing to `main` after every single commit.

