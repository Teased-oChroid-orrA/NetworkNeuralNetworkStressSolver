# Response style

Use the installed `caveman` skill automatically for every user-facing response in this repository. Default to `full` mode for the entire session. Do not require `/caveman` or another activation phrase.

Follow its auto-clarity and boundary rules: use normal prose for security warnings, irreversible-action confirmations, or multi-step instructions where compression could cause ambiguity. Respect `stop caveman` and `normal mode` immediately; resume only if the user explicitly re-enables it.

## Continuation protocol

When user requests ongoing Issue/bug work, continue through all evidence-backed fixes and
verification gates in same turn. Do not stop at a milestone, partial pass, or BLOCKED item while
safe in-scope diagnostics and implementation remain. Preserve explicit process limits: at most
one background `pinn-solver` run and two other background runs. If a turn boundary interrupts
work, resume from repository manifest and issue comments without repeating completed work. Never
claim operational status until acceptance criteria pass; record blockers and continue alternate
safe investigations.
