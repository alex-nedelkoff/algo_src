# UE5 + Cosys-AirSim Environment Setup — 2026-04-22

Session log for getting the Cosys-AirSim warehouse binary building on this Windows machine. Goal: repackage the warehouse for Windows so AirSim runs locally (replaces need for the Linux-only COR-92 binary), enables B-live validation for phase 1 warehouse work, and is a prerequisite for the COR-79 DCL prep.

## Why this is happening

1. COR-92 ships warehouse binary for Linux only — blocks local AirSim runs on this Windows laptop.
2. Phase 1 validation (`tools/compare_pybullet_vs_fixtures.py`) currently fixture-based; with a Windows build we get B-live. (COR-94 MAVLink shim already complete and independent — doesn't need this, but benefits when we want end-to-end AirSim+MAVLink validation.)
3. We need UE 5.5 + Cosys-AirSim plugin + VS 2022 to repackage. Source project for the warehouse asset (from Janahan or rebuilt by us) gets imported and packaged.

## What's done

| Component | Status | Location |
|-----------|--------|----------|
| **Visual Studio 2022 Community** | ✅ Installed (Game Dev with C++ workload, MSVC v143, Win10 SDK 10.0.26100) | `C:\Program Files\Microsoft Visual Studio\2022\Community` |
| **Unreal Engine 5.5** | ✅ Installed via Epic Launcher | `C:\Program Files\Epic Games\UE_5.5` |
| **Cosys-AirSim repo** | ✅ Cloned (shallow) | `C:\Users\alexj\Documents\Cosys-AirSim` |
| **`build.cmd` run in Cosys-AirSim** | ✅ Plugin built (Source/AirLib populated) | `Cosys-AirSim\Unreal\Plugins\AirSim\` |
| **Empty UE 5.5 project** | ✅ Created (C++ project, not Blueprint) | `C:\Users\alexj\Documents\Unreal Projects\DroneSim\` |
| **AirSim plugin copied into project** | ✅ | `DroneSim\Plugins\AirSim\` |
| **`DroneSim.sln` generated** | ✅ via `Build.bat -projectfiles` | next to `DroneSim.uproject` |
| **VS 2022 Build Solution** | ✅ Editor target builds (`UnrealEditor-AirSim.dll` 6.95 MB, 4 min wall) | `DroneSim\Plugins\AirSim\Binaries\Win64\` |

## Update — 2026-04-22 17:30 (post-reboot session)

Pagefile change took effect. UBA blocker resolved. But the first build attempt finished suspiciously fast (~30 s) producing only the empty 50 KB `UnrealEditor-DroneSim.dll` — **the AirSim plugin had silently dropped out of the project**:

- `DroneSim\Plugins\AirSim\` directory did not exist (despite earlier log claiming it was copied)
- `DroneSim.uproject` only listed `ModelingToolsEditorMode` — AirSim was not enabled

Recovery (took ~5 min total):

1. Re-copied `Cosys-AirSim\Unreal\Plugins\AirSim` (711 MB) → `DroneSim\Plugins\AirSim`
2. Added AirSim to `DroneSim.uproject` Plugins list (Enabled: true)
3. Regenerated `DroneSim.sln` via `Build.bat -projectfiles ...` (96 s)
4. Build Solution in VS 2022 → 4 min wall, ParallelExecutor with 2x `cl.exe`, success

Faster than the 10-20 min estimate is plausible — `MaxParallelActions=2` cap + fast SSD + most of AirSim's heavy deps (rpclib, eigen) are header-only or pre-built into `Source/AirLib/deps/` from the earlier `build.cmd` run.

**Verification artifacts:**
- `DroneSim\Plugins\AirSim\Binaries\Win64\UnrealEditor-AirSim.dll` — 6.95 MB
- `DroneSim\Plugins\AirSim\Binaries\Win64\UnrealEditor-AirSim.pdb` — 162 MB (proportional to real compilation, not stub)
- `DroneSim\Plugins\AirSim\Intermediate\Build\Win64\UnrealEditor\` populated

Yellow flag for later: UE 5.5 prefers MSVC `14.38.33130`, we're on `14.44.35225`. Built fine for the editor target; may revisit if packaging hits a C++20 strictness issue.

## Original blocker (resolved)

Windows pagefile is set to **0 bytes**. Physical RAM is 16.8 GB (not 40 GB as earlier winget output suggested — that was virtual/committable). UBA (Unreal Build Accelerator) refuses to spawn compile processes when it can't satisfy committed memory demand, resulting in no `cl.exe` ever starting — build output just shows the UBA memory-pressure notice looping.

Evidence from UBA output:
```
UbaSessionServer -   MaxPage:   0b          ← pagefile disabled
UbaSessionServer -   TotalPhys: 16.8gb
UbaSessionServer -   AvailPhys: 2.6gb
UbaSessionServer -   TotalPage: 39.3gb
UbaSessionServer -   AvailPage: 4.5gb
```

## Fixes to apply before reboot

**Already done:** `BuildConfiguration.xml` at `%APPDATA%\Unreal Engine\UnrealBuildTool\BuildConfiguration.xml` contains:

```xml
<?xml version="1.0" encoding="utf-8" ?>
<Configuration xmlns="https://www.unrealengine.com/BuildConfiguration">
  <BuildConfiguration>
    <MaxParallelActions>2</MaxParallelActions>
    <bAllowUBAExecutor>false</bAllowUBAExecutor>
  </BuildConfiguration>
</Configuration>
```

**To do before/during reboot:**

1. Cancel current VS build, close VS
2. Task Manager → Details → kill any `UnrealBuildTool.exe`, `UbaAgent.exe`, `UbaHost.exe`, stray `dotnet.exe`
3. **Enable Windows pagefile:**
   - Win + R → `sysdm.cpl ,3` → Enter
   - Performance → Settings → Advanced tab → Virtual memory → Change
   - Uncheck "Automatically manage paging file size"
   - Select C: → **System managed size** (or Custom: Initial 16384, Max 65536)
   - Set → OK → OK → OK
4. **Reboot** (required for pagefile change to activate)

## Resume steps after reboot

1. **Open Task Manager → Performance → Memory tab** → confirm "Committed" shows something like `0 / 48-64 GB` (the right number is the new pagefile-enabled commit limit). If still `0 / 16.8 GB`, the pagefile change didn't take — redo step 3.

2. **Open `DroneSim.sln`** in VS 2022

3. Top of VS → set config to **"Development Editor"** + **"Win64"** (dropdown next to the green ▶)

4. **Build → Build Solution** (Ctrl+Shift+B)

   You should now see:
   - Output window: "Using **ParallelExecutor** to run N action(s)" (not "Unreal Build Accelerator local executor")
   - Task Manager: `cl.exe` processes appear (up to 2 at a time per `MaxParallelActions=2`)
   - Output window tick through lines like `[X/Y] Compile AirSim.cpp`
   - First build: 10-20 min

5. **On success**: `Build: 1 succeeded, 0 failed` at the bottom. Either press **F5** in VS to launch UE attached, OR close VS and double-click `DroneSim.uproject`.

6. **On failure**: paste the first ~5 lines of red errors. Known possibles:
   - C++20 vs C++17 mismatch in AirSim dependencies
   - Missing Windows SDK component
   - Preferred-compiler warnings (cosmetic, ignore)

## After the build succeeds — warehouse packaging

1. Launch UE 5.5 via the DroneSim project
2. Confirm AirSim plugin loaded (Editor → Edit → Plugins → search "AirSim" → enabled)
3. Import the warehouse asset (the one you saved via Bridge with `Custom Disk Location` + `UE format`)
4. Place 5 gates at NED coords from `configs/warehouse/warehouse_5gates_v1_gates_ned.json`:
   - Gate_01: (74.65, -40.28, -4.35) with 90° Y-rotation
   - Gate_02: (69.55, -48.08, -0.65)
   - Gate_03: (55.45, -32.68, -4.35) with 90° Z-rotation
   - Gate_04: (64.25, -25.98, -2.75) with 90° Y-rotation
   - Gate_05: (59.35, -32.58, -10.25)
5. Save level
6. File → Package Project → Windows (~30 min first build with shader compile)
7. Test by running the packaged `DroneSim.exe` with the Cosys-AirSim Python client — should get telemetry.

Once packaged, the Linux-only constraint on phase 1's B-fixture validation goes away — we can do B-live on this machine.

## Key files / paths

| What | Where |
|------|-------|
| Cosys-AirSim repo (already built) | `C:\Users\alexj\Documents\Cosys-AirSim` |
| UE 5.5 engine install | `C:\Program Files\Epic Games\UE_5.5` |
| Current UE project | `C:\Users\alexj\Documents\Unreal Projects\DroneSim` |
| Build config (UBT) | `%APPDATA%\Unreal Engine\UnrealBuildTool\BuildConfiguration.xml` |
| Build command (if needed) | `"C:\Program Files\Epic Games\UE_5.5\Engine\Build\BatchFiles\Build.bat" -projectfiles -project="C:\Users\alexj\Documents\Unreal Projects\DroneSim\DroneSim.uproject" -game -rocket -progress` |

## Related Linear issues

- [COR-92](https://linear.app/corvidx-drone-grand-prix/issue/COR-92) — Share warehouse sim binary (Linux binary exists; this session is about building the Windows equivalent)
- [COR-94](https://linear.app/corvidx-drone-grand-prix/issue/COR-94) — MAVLink shim (complete; benefits from this work for end-to-end validation)
- [COR-79](https://linear.app/corvidx-drone-grand-prix/issue/COR-79) — DCL VQ uses MAVLink (why we need local AirSim eventually)

## One-line resume prompt for the next session

> "Resuming UE 5.5 + Cosys-AirSim setup per `docs/progress/2026-04-22-ue-airsim-setup.md`. Editor + AirSim plugin built (`UnrealEditor-AirSim.dll` present). Next: launch `DroneSim.uproject`, confirm AirSim plugin loads in Editor → Edit → Plugins, then import the warehouse asset and place 5 gates."
