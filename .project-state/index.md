# Project State Index

## Current Focus
- Goal: maintain TRELLIS-new project state under the updated one-entrypoint-per-EXE ledger contract
- Active thread: FaceScape training/configuration, data processing, and GT reconstruction audit state
- Last updated: 2026-07-17
- Last full scan: 2026-07-17
- Language: zh
- Archive policy: user-directed

## Read Routing
| Situation | Required read order | Conditional targeted reads | Write targets when state changes |
|---|---|---|---|
| Continue active work or plan | `current.md` | IDs in `Relevant State`; `timeline.md` only if current is stale or conflicts | `current.md`, `index.md`; `timeline.md` only for a material event |
| User reports or agent performs a project execution | `runs.md`, target `executables.md` section | Referenced `CFG` and `ART` sections; inspect a run directory only for required facts | `runs.md`; new/changed `executables.md`, `experiment-configs.md`, `artifacts.md`; update `current.md` and `index.md` if context changed |
| Analyze an existing result | target `RUN` or `ART` section | Linked `ART`, `RUN`, `CFG`, and `EXE` sections needed to evaluate the result | `runs.md` for analysis; `current.md` and `timeline.md` only if conclusion changes next action |
| Invoke, inventory, or change a direct entry point | target `executables.md` section | Referenced `CFG` sections and current state only when needed | `executables.md`; `current.md` and `timeline.md` for a material interface or behavior change |
| Locate a concrete resource or its provenance | target `artifacts.md` section | Producing `RUN` section only when `Produced by run` has an ID | `artifacts.md` only when the resource record changes |
| Locate or register a configuration file | target `experiment-configs.md` section | Referenced `EXE` section only when consumption details matter | `experiment-configs.md`; `executables.md` if its external contract changed |
| Recover a previous active state | target `history.md` snapshot | Current and detailed linked records only to resolve conflicts | `current.md`, then `index.md` after recovery |
| First use, full scan, or ledger gap check | `executables.md`, `experiment-configs.md`, `artifacts.md` description lists | Static repository inventory; exact existing `EXE`, `CFG`, and `ART` sections; `runs.md` only with explicit execution evidence | Add/update `executables.md`, `experiment-configs.md`, `artifacts.md`, and evidence-backed `runs.md`; then `current.md`, `index.md`; `timeline.md` only for a material finding |
| No durable project knowledge | none | none | none |

## Recent Anchors
- Timeline: EVT-20260717-000000-04
- Runs: none
- Config files: CFG-20260717-103, CFG-20260717-106, CFG-20260717-108, CFG-20260717-111
- Executables: EXE-20260717-105, EXE-20260717-130, EXE-20260717-141, EXE-20260717-144
- Artifacts: ART-20260717-001, ART-20260717-002
