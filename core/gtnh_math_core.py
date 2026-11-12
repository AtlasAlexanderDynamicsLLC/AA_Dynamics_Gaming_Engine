"""
Deterministic GTNH 2.9.0 Telemetry, Ore-Grid, Overclock & Explosion-Safety Core.
Zero external dependencies; safe for high-frequency tick evaluation.
"""

import math
from typing import Dict, List, Tuple

GT_VOLTAGE_TIERS: List[Tuple[str, int]] = [
    ("ULV", 8),
    ("LV", 32),
    ("MV", 128),
    ("HV", 512),
    ("EV", 2048),
    ("IV", 8192),
    ("LuV", 32768),
    ("ZPM", 131072),
    ("UV", 524288),
    ("UHV", 2097152),
]

TIER_NAME_TO_EU: Dict[str, int] = dict(GT_VOLTAGE_TIERS)


def resolve_tier_name(eu_t: int) -> str:
    for name, max_eu in GT_VOLTAGE_TIERS:
        if eu_t <= max_eu:
            return name
    return "UEV+"


def minecraft_bearing(dx: float, dz: float) -> str:
    """
    Converts block deltas (target - player) into Minecraft compass directions.
    In Minecraft: -Z is North, +Z is South, +X is East, -X is West.
    """
    if abs(dx) < 0.5 and abs(dz) < 0.5:
        return "HERE"
    # Angle clockwise from North (0 deg = -Z, 90 deg = +X, 180 deg = +Z, 270 deg = -X)
    angle = (math.degrees(math.atan2(dx, -dz)) + 360.0) % 360.0
    compass = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((angle + 22.5) // 45.0) % 8
    return f"{compass[idx]} ({int(round(angle))}°)"


def resolve_nearest_ore_center(block_x: int, block_z: int) -> Dict[str, int | bool | float | str]:
    """Maps player block coordinates to the nearest valid GTNH 3x3 ore-vein chunk center."""
    chunk_x = block_x // 16
    chunk_z = block_z // 16
    is_in_ore_chunk = (abs(chunk_x) % 3 == 1) and (abs(chunk_z) % 3 == 1)

    min_distance = math.inf
    target_chunk_x = 0
    target_chunk_z = 0
    target_block_x = 0
    target_block_z = 0

    for cx in range(chunk_x - 3, chunk_x + 4):
        if abs(cx) % 3 != 1:
            continue
        for cz in range(chunk_z - 3, chunk_z + 4):
            if abs(cz) % 3 != 1:
                continue
            bx = cx * 16 + 8
            bz = cz * 16 + 8
            dist = math.hypot(bx - block_x, bz - block_z)
            if dist < min_distance or (dist == min_distance and (cx, cz) < (target_chunk_x, target_chunk_z)):
                min_distance = dist
                target_chunk_x, target_chunk_z = cx, cz
                target_block_x, target_block_z = bx, bz

    dx = target_block_x - block_x
    dz = target_block_z - block_z

    return {
        "current_chunk_x": chunk_x,
        "current_chunk_z": chunk_z,
        "is_in_ore_chunk": is_in_ore_chunk,
        "target_chunk_x": target_chunk_x,
        "target_chunk_z": target_chunk_z,
        "target_block_x": target_block_x,
        "target_block_z": target_block_z,
        "delta_x": dx,
        "delta_z": dz,
        "compass": minecraft_bearing(dx, dz),
        "distance_blocks": round(min_distance, 2),
    }


def calculate_gt_overclock(
    base_eu_t: int,
    base_ticks: int,
    machine_tier_eu_t: int,
    perfect_overclock: bool = False,
) -> Dict[str, int | float | str]:
    """Computes deterministic GregTech recipe overclocking clamped at the 1-tick floor."""
    if base_eu_t <= 0 or base_ticks <= 0 or machine_tier_eu_t <= 0:
        raise ValueError("base_eu_t, base_ticks, and machine_tier_eu_t must be > 0.")
    if base_eu_t > machine_tier_eu_t:
        raise ValueError(f"Insufficient tier: recipe needs {base_eu_t} EU/t, machine max is {machine_tier_eu_t} EU/t.")

    scaled_eu_t = base_eu_t
    scaled_ticks = base_ticks
    overclock_count = 0
    divisor = 4 if perfect_overclock else 2

    while scaled_eu_t * 4 <= machine_tier_eu_t and scaled_ticks > 1:
        scaled_eu_t *= 4
        scaled_ticks = max(1, scaled_ticks // divisor)
        overclock_count += 1

    total_eu = scaled_eu_t * scaled_ticks
    return {
        "recipe_min_tier": resolve_tier_name(base_eu_t),
        "machine_tier": resolve_tier_name(machine_tier_eu_t),
        "overclock_count": overclock_count,
        "final_eu_t": scaled_eu_t,
        "final_ticks": scaled_ticks,
        "duration_sec": round(scaled_ticks / 20.0, 2),
        "total_eu": total_eu,
    }


def evaluate_safety_hazards(telemetry: Dict) -> List[Dict[str, str]]:
    """Deterministic sub-millisecond check for catastrophic GTNH base hazards."""
    alerts: List[Dict[str, str]] = []

    if (telemetry.get("is_raining") or telemetry.get("is_thundering")) and telemetry.get("exposed_gt_machines", 0) > 0:
        cnt = telemetry["exposed_gt_machines"]
        alerts.append({
            "code": "RAIN_EXPOSURE",
            "severity": "CRITICAL",
            "message": f"Rain/storm active with {cnt} uncovered GT machine(s) — Cover top faces immediately or machines will explode!"
        })

    mach_max = int(telemetry.get("crosshair_machine_max_eu_t") or 0)
    cable_eu = int(telemetry.get("crosshair_cable_eu_t") or 0)
    if mach_max > 0 and cable_eu > mach_max:
        alerts.append({
            "code": "OVER_VOLTAGE",
            "severity": "CRITICAL",
            "message": f"OVER-VOLTAGE: {cable_eu} EU/t ({resolve_tier_name(cable_eu)}) line exceeds {mach_max} EU/t ({resolve_tier_name(mach_max)}) machine hull!"
        })

    boiler_temp = float(telemetry.get("boiler_temp_c") or 0.0)
    boiler_water = int(telemetry.get("boiler_water_mb") or -1)
    if boiler_temp >= 100.0 and boiler_water == 0:
        alerts.append({
            "code": "DRY_BOILER",
            "severity": "CRITICAL",
            "message": f"DRY STEAM BOILER ({boiler_temp:.0f}°C, 0 mB water) — DO NOT pipe water in! Break boiler or cool below 100°C first!"
        })

    srv = telemetry.get("server_factory") or {}
    if int(srv.get("dry_boilers_count") or 0) > 0:
        alerts.append({
            "code": "CROSS_DIM_BOILER",
            "severity": "CRITICAL",
            "message": f"REMOTE SERVER ALERT: {srv['dry_boilers_count']} chunkloaded dry steam boiler(s) >=100°C with 0 water!"
        })

    maint = srv.get("maintenance_alerts") or []
    if maint:
        m0 = maint[0]
        alerts.append({
            "code": "CROSS_DIM_MAINT",
            "severity": "WARNING",
            "message": f"MAINTENANCE FAULT ({len(maint)} total): {m0.get('name')} in Dim {m0.get('dim')} at ({m0.get('x')}, {m0.get('y')}, {m0.get('z')})"
        })

    if int(srv.get("eu_capacity_total") or 0) > 100000 and float(srv.get("eu_fill_percent") or 100.0) < 10.0:
        alerts.append({
            "code": "LOW_GRID_EU",
            "severity": "CRITICAL",
            "message": f"FACTORY POWER DRAIN: Global EU buffer at {srv.get('eu_fill_percent')}% — risk of multiblock recipe voiding!"
        })

    return alerts


def run_self_tests() -> None:
    r1 = resolve_nearest_ore_center(-20, -20)
    assert r1["current_chunk_x"] == -2 and r1["is_in_ore_chunk"] is False
    assert (r1["target_chunk_x"], r1["target_chunk_z"]) == (-1, -1)
    assert (r1["target_block_x"], r1["target_block_z"]) == (-8, -8)
    assert r1["distance_blocks"] == 16.97, f"Expected 16.97, got {r1['distance_blocks']}"

    r2 = resolve_nearest_ore_center(24, -72)
    assert (r2["current_chunk_x"], r2["current_chunk_z"]) == (1, -5)
    assert r2["is_in_ore_chunk"] is False
    assert (r2["target_chunk_x"], r2["target_chunk_z"]) == (1, -4)
    assert (r2["target_block_x"], r2["target_block_z"]) == (24, -56)
    assert r2["distance_blocks"] == 16.0

    oc1 = calculate_gt_overclock(30, 200, 512, perfect_overclock=False)
    assert (oc1["overclock_count"], oc1["final_eu_t"], oc1["final_ticks"], oc1["total_eu"]) == (2, 480, 50, 24000)

    oc2 = calculate_gt_overclock(16, 4, 2048, perfect_overclock=True)
    assert (oc2["overclock_count"], oc2["final_eu_t"], oc2["final_ticks"], oc2["total_eu"]) == (1, 64, 1, 64)


if __name__ == "__main__":
    run_self_tests()
    print("✅ GTNH 2.9.0 Math & Safety Core passed all verification assertions.")
