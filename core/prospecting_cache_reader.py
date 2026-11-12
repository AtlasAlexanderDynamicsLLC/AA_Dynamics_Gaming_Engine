import os
import re
import gzip
import json
import math
import struct
from io import BytesIO
from typing import Any

GTNH_VEIN_CATALOG = {
    "ore.mix.magnetite":      {"label": "Magnetite",          "ores": "Magnetite, Iron, Vanadium Magnetite, Gold", "y": "50-120"},
    "ore.mix.chalcopyrite":   {"label": "Chalcopyrite",       "ores": "Chalcopyrite, Iron, Pyrite, Copper", "y": "10-30"},
    "ore.mix.cassiterite":    {"label": "Cassiterite",        "ores": "Tin, Cassiterite", "y": "40-120"},
    "ore.mix.coal":           {"label": "Coal",               "ores": "Coal, Lignite", "y": "50-80"},
    "ore.mix.iron":           {"label": "Limonite/Iron",      "ores": "Brown Limonite, Yellow Limonite, Banded Iron, Malachite", "y": "10-40"},
    "ore.mix.copper":         {"label": "Tetrahedrite",       "ores": "Tetrahedrite, Copper, Stibnite", "y": "80-120"},
    "ore.mix.redstone":       {"label": "Redstone",           "ores": "Redstone, Ruby, Cinnabar", "y": "10-40"},
    "ore.mix.lapis":          {"label": "Lapis",              "ores": "Lapis, Lazurite, Sodalite, Calcite", "y": "20-50"},
    "ore.mix.diamond":        {"label": "Diamond/Graphite",   "ores": "Graphite, Diamond, Coal", "y": "5-20"},
    "ore.mix.quartz":         {"label": "Quartzite",          "ores": "Quartzite, Barite, Certus Quartz", "y": "40-80"},
    "ore.mix.salts":          {"label": "Salt/Rock Salt",     "ores": "Rock Salt, Salt, Lepidolite, Spodumene", "y": "50-70"},
    "ore.mix.apatite":        {"label": "Apatite",            "ores": "Apatite, Tricalcium Phosphate, Pyrochlore", "y": "40-60"},
    "ore.mix.oilsand":        {"label": "Oilsands",           "ores": "Oilsands", "y": "50-80"},
    "ore.mix.nickel":         {"label": "Nickel/Garnierite",  "ores": "Garnierite, Nickel, Cobaltite, Pentlandite", "y": "10-40"},
    "ore.mix.galena":         {"label": "Galena/Silver",      "ores": "Galena, Silver, Lead", "y": "30-60"},
    "ore.mix.bauxite":        {"label": "Bauxite",            "ores": "Bauxite, Aluminium, Ilmenite", "y": "50-90"},
    "ore.mix.mica":           {"label": "Mica",               "ores": "Kyanite, Mica, Cassiterite, Pollucite", "y": "20-40"},
    "ore.mix.manganese":      {"label": "Manganese",          "ores": "Grossular, Spessartine, Pyrolusite, Tantalite", "y": "20-30"},
    "ore.mix.monazite":       {"label": "Monazite",           "ores": "Bastnasite, Monazite, Neodymium", "y": "20-40"},
    "ore.mix.Molybdenum":     {"label": "Molybdenum",         "ores": "Wulfenite, Molybdenite, Molybdenum, Powellite", "y": "20-50"},
    "ore.mix.beryllium":      {"label": "Beryllium",          "ores": "Beryllium, Emerald, Thorium", "y": "5-30"},
    "ore.mix.sulfur":         {"label": "Sulfur",             "ores": "Sulfur, Pyrite, Sphalerite", "y": "5-20"},
    "ore.mix.saltpeter":      {"label": "Saltpeter",          "ores": "Saltpeter, Diatomite, Electrotine, Alunite", "y": "10-40"},
    "ore.mix.sapphire":       {"label": "Sapphire",           "ores": "Almandine, Pyrope, Sapphire, Green Sapphire", "y": "10-40"},
    "ore.mix.olivine":        {"label": "Olivine/Bentonite",  "ores": "Bentonite, Magnesite, Olivine, Glauconite", "y": "10-40"},
    "ore.mix.pitchblende":    {"label": "Pitchblende",        "ores": "Pitchblende, Uraninite", "y": "10-40"},
    "ore.mix.platinum":       {"label": "Platinum/Sheldonite","ores": "Cooperite, Palladium, Platinum, Iridium", "y": "40-50"},
    "ore.mix.tungstate":      {"label": "Tungstate",          "ores": "Scheelite, Tungstate, Lithium", "y": "20-50"},
    "ore.mix.naquadah":       {"label": "Naquadah",           "ores": "Naquadah, Enriched Naquadah", "y": "10-60"},
}

def _compass_from_delta(dx: int, dz: int) -> str:
    if dx == 0 and dz == 0:
        return "HERE (0°)"
    deg = (math.degrees(math.atan2(dx, -dz)) + 360.0) % 360.0
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    card = dirs[int((deg + 22.5) // 45.0) % 8]
    return f"{card} ({round(deg)}°)"

def _parse_dim_id(dim_value: Any) -> int:
    if isinstance(dim_value, int):
        return dim_value
    s = str(dim_value or "0")
    m = re.search(r"\((-?\d+)\)", s)
    if m:
        return int(m.group(1))
    m2 = re.search(r"(-?\d+)", s)
    return int(m2.group(1)) if m2 else 0

def _enrich_vein_name(raw_name: str) -> dict[str, str]:
    clean = (raw_name or "").strip()
    meta = GTNH_VEIN_CATALOG.get(clean) or GTNH_VEIN_CATALOG.get(clean.lower())
    if meta:
        return {"raw_key": clean, "display_name": meta["label"], "ores": meta["ores"], "y_range": meta["y"]}
    short = clean
    for prefix in ("ore.mix.custom.", "ore.mix.", "bw.ore.mix."):
        if short.lower().startswith(prefix):
            short = short[len(prefix):]
            break
    short_label = short.replace("_", " ").replace(".", " ").title() or "Unknown Vein"
    return {"raw_key": clean, "display_name": short_label, "ores": short_label, "y_range": "Varies"}

class ProspectingCacheIndex:
    def __init__(self, gtnh_instance_dir: str | None = None):
        self.instance_dir = gtnh_instance_dir or ""
        self._dir_signature: tuple = ()
        self.veins: list[dict[str, Any]] = []
        self.waypoints: list[dict[str, Any]] = []
        self._server_veins: dict[tuple[int, int, int], dict[str, Any]] = {}

    def set_instance_dir(self, path: str):
        if path and path != self.instance_dir:
            self.instance_dir = path
            self._dir_signature = ()

    def ingest_server_veins(self, generated_veins: list[dict[str, Any]]):
        if not generated_veins:
            return
        for gv in generated_veins:
            raw_key = str(gv.get("raw_key") or "")
            if not raw_key or raw_key == "ore.mix.none":
                continue
            dim = int(gv.get("dim", 0))
            cx = int(gv.get("chunk_x", 0))
            cz = int(gv.get("chunk_z", 0))
            key = (dim, cx, cz)
            if key not in self._server_veins:
                meta = _enrich_vein_name(raw_key)
                self._server_veins[key] = {
                    "dim": dim,
                    "chunk_x": cx,
                    "chunk_z": cz,
                    "x": int(gv.get("x", cx * 16 + 8)),
                    "z": int(gv.get("z", cz * 16 + 8)),
                    "raw_key": meta["raw_key"],
                    "name": meta["display_name"],
                    "ores": meta["ores"],
                    "y_range": meta["y_range"],
                    "depleted": False,
                    "source": "ServerCache (Live)",
                }

    def refresh_if_needed(self, force: bool = False):
        if not self.instance_dir or not os.path.isdir(self.instance_dir):
            return
        jm_root = os.path.join(self.instance_dir, "journeymap", "data")
        entries = []
        if os.path.isdir(jm_root):
            for dirpath, _, filenames in os.walk(jm_root):
                for fn in filenames:
                    if fn.lower().endswith(".json"):
                        fp = os.path.join(dirpath, fn)
                        try:
                            st = os.stat(fp)
                            entries.append((fp, int(st.st_mtime), st.st_size))
                        except OSError:
                            pass
        sig = tuple(sorted(entries))
        if not force and sig == self._dir_signature:
            return
        self._dir_signature = sig
        self.waypoints.clear()
        seen_wp = set()
        if os.path.isdir(jm_root):
            for dirpath, _, filenames in os.walk(jm_root):
                if os.path.basename(dirpath).lower() != "waypoints":
                    continue
                for fn in filenames:
                    if not fn.lower().endswith(".json"):
                        continue
                    try:
                        with open(os.path.join(dirpath, fn), "r", encoding="utf-8", errors="replace") as f:
                            data = json.load(f)
                        name = str(data.get("name") or fn[:-5]).strip()
                        bx, by, bz = int(data.get("x", 0)), int(data.get("y", 64)), int(data.get("z", 0))
                        dims = data.get("dimensions") or [0]
                        dim = int(dims[0]) if dims else 0
                        if (dim, bx, bz, name) in seen_wp:
                            continue
                        seen_wp.add((dim, bx, bz, name))
                        self.waypoints.append({
                            "dim": dim, "x": bx, "y": by, "z": bz,
                            "chunk_x": bx >> 4, "chunk_z": bz >> 4,
                            "name": name, "category": "WAYPOINT",
                        })
                    except Exception:
                        pass

    def query_near_player(self, player_x: int, player_z: int, dimension_val: Any = 0, search_filter: str = "", max_veins: int = 12, max_waypoints: int = 8) -> dict[str, Any]:
        self.refresh_if_needed()
        dim_id = _parse_dim_id(dimension_val)
        q = (search_filter or "").strip().lower()

        all_veins = list(self._server_veins.values())
        matched_veins = []
        for v in all_veins:
            if v["dim"] != dim_id:
                continue
            if q and q not in v["name"].lower() and q not in v["ores"].lower() and q not in v["raw_key"].lower():
                continue
            dx = int(v["x"]) - int(player_x)
            dz = int(v["z"]) - int(player_z)
            entry = dict(v)
            entry["distance_blocks"] = round(math.hypot(dx, dz), 1)
            entry["delta_x"] = dx
            entry["delta_z"] = dz
            entry["compass"] = _compass_from_delta(dx, dz)
            matched_veins.append(entry)

        matched_veins.sort(key=lambda item: item["distance_blocks"])

        matched_wps = []
        for w in self.waypoints:
            if w["dim"] != dim_id:
                continue
            if q and q not in w["name"].lower():
                continue
            dx = int(w["x"]) - int(player_x)
            dz = int(w["z"]) - int(player_z)
            entry = dict(w)
            entry["distance_blocks"] = round(math.hypot(dx, dz), 1)
            entry["compass"] = _compass_from_delta(dx, dz)
            matched_wps.append(entry)

        matched_wps.sort(key=lambda item: item["distance_blocks"])

        return {
            "dimension_id": dim_id,
            "filter": q,
            "total_veins_cached": len(all_veins),
            "dim_veins_matched": len(matched_veins),
            "total_waypoints_cached": len(self.waypoints),
            "nearest_veins": matched_veins[:max_veins],
            "nearest_waypoints": matched_wps[:max_waypoints],
        }
