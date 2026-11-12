import os
import re
import json
from datetime import datetime
from PySide6 import QtCore, QtGui, QtWidgets

ENGINE_ROOT = r"E:\SovereignEngine\AA_Dynamics_Gaming_Engine"
TASKS_JSON = os.path.join(ENGINE_ROOT, "config", "active_tasks.json")

TASK_CATEGORIES = {
    "⛏️ Mining & Extraction": {
        "color": "#38BDF8",
        "lead_specialist": "Geologist / Navigator",
        "keywords": ("mine", "mining", "ore", "vein", "dig", "prospect", "copper", "tin", "cassiterite", "chalcopyrite", "tetrahedrite", "iron", "coal", "redstone", "diamond", "bauxite", "galena", "lapis", "quartz", "salt"),
    },
    "⚡ Circuits & Electronics": {
        "color": "#E879F9",
        "lead_specialist": "Chief Industrial Planner",
        "keywords": ("circuit", "board", "vacuum tube", "resistor", "diode", "transistor", "wire", "cable", "red alloy", "solder", "silicon", "wafer"),
    },
    "🏭 Machines & Metallurgy": {
        "color": "#FFB000",
        "lead_specialist": "Chief Industrial Planner",
        "keywords": ("blast furnace", "ebf", "boiler", "steam", "macerator", "bender", "wiremill", "lathe", "assembler", "bronze", "steel", "wrought", "alloy", "hull", "motor", "piston"),
    },
    "🗡️ Weapons, Tools & Armor": {
        "color": "#39FF14",
        "lead_specialist": "Weapon / Armor / Tool Specialist",
        "keywords": ("pickaxe", "hammer", "sword", "cleaver", "crossbow", "armor", "shield", "tinker", "tool rod", "modifier", "repair"),
    },
    "🧭 Exploration & Wayfinding": {
        "color": "#FACC15",
        "lead_specialist": "Wayfinder / Map Specialist",
        "keywords": ("explore", "waypoint", "biome", "village", "temple", "portal", "nether", "twilight", "base", "return"),
    },
}

# Built-in GTNH 2.9.0 Critical-Path Dependency Templates
GTNH_DEPENDENCY_TEMPLATES = [
    (
        ("circuit", "board", "electronic"),
        "⚡ Circuits & Electronics",
        [
            "Mine 500 Copper Ore (Chalcopyrite / Tetrahedrite vein)",
            "Mine 200 Tin & Redstone Ore (Cassiterite & Cinnabar/Redstone)",
            "Tap Rubber Trees for Sticky Resin -> Smelt Raw Rubber Pulp",
            "Craft 1x Copper Cables & Red Alloy Wires (Wiremill / Forge Hammer)",
            "Assemble Coated Circuit Boards + Vacuum Tubes + Resistors",
        ],
    ),
    (
        ("ebf", "electric blast furnace", "blast furnace"),
        "🏭 Machines & Metallurgy",
        [
            "Mine 600 Iron & Copper Ore for LV Machine Hulls & Cupronickel Coils",
            "Mine Nickel / Garnierite Vein for 16x Cupronickel Coil Blocks",
            "Batch-craft 4x LV Energy Hatches, Maintenance Hatch & Muffler Hatch",
            "Assemble 3x3x4 Electric Blast Furnace Multiblock",
        ],
    ),
    (
        ("bronze", "steam age", "boiler"),
        "🏭 Machines & Metallurgy",
        [
            "Mine 400 Copper Ore + 150 Tin / Cassiterite Ore (3:1 Bronze Ratio)",
            "Gather Clay, Sand & Gravel for Coke Oven Bricks & Firebricks",
            "Build Steam Boiler + Steam Macerator, Forge Hammer & Alloy Smelter",
        ],
    ),
]


class TaskManager:
    """Manages categorized modpack tasks, sub-task dependencies, and active focus filtering."""

    def __init__(self):
        self.tasks: list[dict] = []
        self._next_id = 1
        self.load()

    def load(self):
        if os.path.isfile(TASKS_JSON):
            try:
                with open(TASKS_JSON, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.tasks = data.get("tasks", [])
                self._next_id = int(data.get("next_id", len(self.tasks) + 1))
                return
            except Exception:
                pass
        # Default starter task list if none exists
        self.add_task("Mine 500 Copper & Cassiterite Ore for LV Circuits", auto_expand=True)

    def save(self):
        os.makedirs(os.path.dirname(TASKS_JSON), exist_ok=True)
        try:
            with open(TASKS_JSON, "w", encoding="utf-8") as f:
                json.dump({"next_id": self._next_id, "tasks": self.tasks}, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def classify_category(self, title: str) -> tuple[str, str]:
        low = title.lower()
        # If it mentions mining/ore explicitly, prioritize Mining & Extraction so Geologist leads
        if any(w in low for w in ("mine ", "mining ", "500 ", "ore", "vein", "dig ")):
            return "⛏️ Mining & Extraction", "Geologist / Navigator"
        for cat, spec in TASK_CATEGORIES.items():
            if any(k in low for k in spec["keywords"]):
                return cat, spec["lead_specialist"]
        return "⛏️ Mining & Extraction", "Geologist / Navigator"

    def add_task(self, title: str, category: str = None, subtasks: list[str] = None, auto_expand: bool = True) -> dict:
        clean_title = title.strip()
        cat, specialist = self.classify_category(clean_title)
        if category and category in TASK_CATEGORIES:
            cat = category
            specialist = TASK_CATEGORIES[cat]["lead_specialist"]

        expanded_subs = list(subtasks or [])
        if auto_expand and not expanded_subs:
            low = clean_title.lower()
            for trig_words, tmpl_cat, tmpl_subs in GTNH_DEPENDENCY_TEMPLATES:
                if any(w in low for w in trig_words):
                    expanded_subs = list(tmpl_subs)
                    break
            if not expanded_subs and ("mine" in low or "ore" in low):
                ore_m = re.search(r"(\d+)\s+([a-zA-Z\s]+?)\s*ore", low)
                if ore_m:
                    qty, ore_type = ore_m.group(1), ore_m.group(2).strip().title()
                    expanded_subs = [
                        f"Locate nearest {ore_type} 3x3 vein center on Geologist Radar",
                        f"Sink vertical shaft to target Y-level and extract {qty} {ore_type} Ore",
                        f"Macerate & wash {ore_type} Crushed Ore for byproduct yield",
                    ]

        task_obj = {
            "id": self._next_id,
            "title": clean_title,
            "category": cat,
            "lead_specialist": specialist,
            "done": False,
            "active": True,
            "created_at": datetime.now().strftime("%H:%M"),
            "subtasks": [{"text": s, "done": False} for s in expanded_subs],
        }
        # Mark new task as primary active task
        for t in self.tasks:
            t["active"] = False
        self.tasks.insert(0, task_obj)
        self._next_id += 1
        self.save()
        return task_obj

    def get_active_task(self) -> dict | None:
        for t in self.tasks:
            if not t.get("done") and t.get("active"):
                return t
        for t in self.tasks:
            if not t.get("done"):
                t["active"] = True
                return t
        return None

    def set_active_task(self, task_id: int):
        for t in self.tasks:
            t["active"] = (t["id"] == task_id)
        self.save()

    def toggle_task_done(self, task_id: int):
        for t in self.tasks:
            if t["id"] == task_id:
                t["done"] = not t.get("done", False)
                if t["done"] and t.get("active"):
                    t["active"] = False
                break
        self.get_active_task()
        self.save()

    def toggle_subtask_done(self, task_id: int, sub_idx: int):
        for t in self.tasks:
            if t["id"] == task_id:
                subs = t.get("subtasks", [])
                if 0 <= sub_idx < len(subs):
                    subs[sub_idx]["done"] = not subs[sub_idx].get("done", False)
                break
        self.save()

    def clear_completed(self):
        self.tasks = [t for t in self.tasks if not t.get("done")]
        self.get_active_task()
        self.save()

    def get_active_filter_keywords(self) -> str:
        """Extracts ore/material keywords from the active task and its unfinished subtasks."""
        act = self.get_active_task()
        if not act:
            return "cassiterite, copper, tin, bronze, iron"
        blob = act["title"] + " " + " ".join(s["text"] for s in act.get("subtasks", []) if not s.get("done"))
        low = blob.lower()
        ore_Automatic_Map = {
            "copper": "copper, chalcopyrite, tetrahedrite, malachite",
            "tin": "tin, cassiterite",
            "bronze": "copper, chalcopyrite, tetrahedrite, tin, cassiterite",
            "circuit": "copper, chalcopyrite, tin, cassiterite, redstone, cinnabar",
            "iron": "iron, magnetite, brown limonite, yellow limonite, banded iron",
            "steel": "iron, magnetite, coal, lignite",
            "redstone": "redstone, cinnabar, ruby",
            "lead": "lead, galena",
            "silver": "silver, galena",
            "nickel": "nickel, garnierite, pentlandite",
            "aluminium": "aluminium, bauxite, ilmenite",
            "diamond": "diamond, coal, graphite",
            "gold": "gold, magnetite",
        }
        matched = []
        for k, expanded in ore_Automatic_Map.items():
            if k in low:
                matched.append(expanded)
        if matched:
            return ", ".join(matched)
        return act["title"]


class TaskOverlayHUD(QtWidgets.QWidget):
    """Frameless, Always-On-Top In-Game Pop-Up Task Menu toggled via the J or F6 key."""

    taskAddedByUser = QtCore.Signal(str)
    requestAiBreakdown = QtCore.Signal(str)
    activeTaskChanged = QtCore.Signal()

    def __init__(self, task_mgr: TaskManager, parent=None):
        super().__init__(parent)
        self.task_mgr = task_mgr
        self.setWindowTitle("Sovereign Task & Dependency HUD")
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.resize(620, 520)
        self._center_on_screen()
        self._build_ui()

    def _center_on_screen(self):
        scr = QtWidgets.QApplication.primaryScreen()
        if scr:
            geo = scr.availableGeometry()
            self.move(geo.x() + geo.width() - self.width() - 40, geo.y() + 80)

    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        frame = QtWidgets.QFrame()
        frame.setStyleSheet(
            "QFrame { background-color: rgba(6, 9, 19, 238); border: 2px solid #00D2FF; border-radius: 8px; } "
            "QLabel { border: none; background: transparent; color: #F8FAFC; font-family: Consolas; } "
            "QTreeWidget { background-color: #0B101D; color: #E2E8F0; border: 1px solid #1E293B; font-family: Consolas; font-size: 9.5pt; } "
            "QLineEdit { background-color: #0B101D; color: #39FF14; border: 1px solid #38BDF8; padding: 6px; font-family: Consolas; font-size: 9.5pt; }"
        )
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        hdr = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("📋 SOVEREIGN MODPACK TASK & DEPENDENCY BOARD  [Press J to Hide]")
        title.setStyleSheet("color: #00D2FF; font-size: 10.5pt; font-weight: bold;")
        hdr.addWidget(title)
        hdr.addStretch(1)

        close_btn = QtWidgets.QPushButton("✕")
        close_btn.setCursor(QtCore.Qt.PointingHandCursor)
        close_btn.setStyleSheet("background: #1E293B; color: #FF003C; border: 1px solid #FF003C; padding: 2px 8px; font-weight: bold;")
        close_btn.clicked.connect(self.hide)
        hdr.addWidget(close_btn)
        layout.addLayout(hdr)

        self.active_banner = QtWidgets.QLabel()
        self.active_banner.setWordWrap(True)
        self.active_banner.setStyleSheet(
            "background-color: #071F12; color: #39FF14; border: 1px solid #39FF14; border-radius: 4px; padding: 6px 10px; font-size: 9.5pt; font-weight: bold;"
        )
        layout.addWidget(self.active_banner)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Categorized Tasks & Sub-Steps (Double-click to set Active / Check to complete)", "Lead Specialist"])
        self.tree.setColumnWidth(0, 420)
        self.tree.itemChanged.connect(self._on_item_checked)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.tree, 1)

        add_row = QtWidgets.QHBoxLayout()
        self.new_task_edit = QtWidgets.QLineEdit()
        self.new_task_edit.setPlaceholderText("Add task (e.g. 'Mine 500 Copper Ore' or 'Craft 16 LV Circuits')...")
        self.new_task_edit.returnPressed.connect(self._on_add_clicked)
        add_row.addWidget(self.new_task_edit, 1)

        add_btn = QtWidgets.QPushButton("+ ADD TASK")
        add_btn.setCursor(QtCore.Qt.PointingHandCursor)
        add_btn.setStyleSheet("background-color: #39FF14; color: #060913; font-weight: bold; padding: 6px 12px; border: none;")
        add_btn.clicked.connect(self._on_add_clicked)
        add_row.addWidget(add_btn)

        ai_btn = QtWidgets.QPushButton("⚡ AI PLAN")
        ai_btn.setCursor(QtCore.Qt.PointingHandCursor)
        ai_btn.setToolTip("Have Chief Industrial Planner break this goal into efficient GTNH sub-tasks")
        ai_btn.setStyleSheet("background-color: #A855F7; color: #060913; font-weight: bold; padding: 6px 10px; border: none;")
        ai_btn.clicked.connect(self._on_ai_breakdown_clicked)
        add_row.addWidget(ai_btn)

        clr_btn = QtWidgets.QPushButton("CLEAR DONE")
        clr_btn.setCursor(QtCore.Qt.PointingHandCursor)
        clr_btn.setStyleSheet("background-color: #1E293B; color: #CBD5E1; padding: 6px 10px; border: 1px solid #475569;")
        clr_btn.clicked.connect(self._on_clear_done)
        add_row.addWidget(clr_btn)

        layout.addLayout(add_row)
        outer.addWidget(frame)
        self.refresh_tree()

    def _on_add_clicked(self):
        txt = self.new_task_edit.text().strip()
        if not txt:
            return
        self.new_task_edit.clear()
        self.taskAddedByUser.emit(txt)

    def _on_ai_breakdown_clicked(self):
        txt = self.new_task_edit.text().strip()
        if not txt:
            act = self.task_mgr.get_active_task()
            txt = act["title"] if act else "LV Circuit Board Production"
        else:
            self.new_task_edit.clear()
        self.requestAiBreakdown.emit(txt)

    def _on_clear_done(self):
        self.task_mgr.clear_completed()
        self.refresh_tree()
        self.activeTaskChanged.emit()

    def _on_item_double_clicked(self, item: QtWidgets.QTreeWidgetItem, col: int):
        tid = item.data(0, QtCore.Qt.UserRole)
        sub_idx = item.data(1, QtCore.Qt.UserRole)
        if tid is not None and sub_idx is None:
            self.task_mgr.set_active_task(int(tid))
            self.refresh_tree()
            self.activeTaskChanged.emit()

    def _on_item_checked(self, item: QtWidgets.QTreeWidgetItem, col: int):
        tid = item.data(0, QtCore.Qt.UserRole)
        sub_idx = item.data(1, QtCore.Qt.UserRole)
        if tid is None:
            return
        is_checked = item.checkState(0) == QtCore.Qt.Checked
        if sub_idx is None:
            for t in self.task_mgr.tasks:
                if t["id"] == int(tid) and bool(t.get("done")) != is_checked:
                    self.task_mgr.toggle_task_done(int(tid))
                    break
        else:
            for t in self.task_mgr.tasks:
                if t["id"] == int(tid):
                    subs = t.get("subtasks", [])
                    if 0 <= int(sub_idx) < len(subs) and bool(subs[int(sub_idx)].get("done")) != is_checked:
                        self.task_mgr.toggle_subtask_done(int(tid), int(sub_idx))
                    break
        QtCore.QTimer.singleShot(10, self.refresh_tree)
        self.activeTaskChanged.emit()

    def refresh_tree(self):
        self.tree.blockSignals(True)
        self.tree.clear()

        act = self.task_mgr.get_active_task()
        if act:
            self.active_banner.setText(
                f"🎯 ACTIVE FOCUS: {act['title']}  |  Category: {act['category']}  |  Enforcer: {act['lead_specialist'].split('/')[0].strip()}"
            )
        else:
            self.active_banner.setText("🎯 NO ACTIVE TASK — Add a task below or say 'Hey Atlas, add mining 500 copper ore to the list'")

        by_cat: dict[str, list[dict]] = {}
        for t in self.task_mgr.tasks:
            by_cat.setdefault(t.get("category", "⛏️ Mining & Extraction"), []).append(t)

        for cat_name, tasks in by_cat.items():
            cat_item = QtWidgets.QTreeWidgetItem(self.tree, [cat_name, ""])
            cat_color = TASK_CATEGORIES.get(cat_name, {}).get("color", "#38BDF8")
            cat_item.setForeground(0, QtGui.QColor(cat_color))
            font = cat_item.font(0)
            font.setBold(True)
            cat_item.setFont(0, font)
            cat_item.setExpanded(True)

            for t in tasks:
                prefix = "★ [ACTIVE] " if (t.get("active") and not t.get("done")) else ""
                t_item = QtWidgets.QTreeWidgetItem(
                    cat_item,
                    [f"{prefix}{t['title']}", t.get("lead_specialist", "").split("/")[0].strip()],
                )
                t_item.setData(0, QtCore.Qt.UserRole, t["id"])
                t_item.setData(1, QtCore.Qt.UserRole, None)
                t_item.setFlags(t_item.flags() | QtCore.Qt.ItemIsUserCheckable)
                t_item.setCheckState(0, QtCore.Qt.Checked if t.get("done") else QtCore.Qt.Unchecked)
                if t.get("active") and not t.get("done"):
                    t_item.setForeground(0, QtGui.QColor("#39FF14"))
                elif t.get("done"):
                    t_item.setForeground(0, QtGui.QColor("#64748B"))

                for idx, sub in enumerate(t.get("subtasks", [])):
                    s_item = QtWidgets.QTreeWidgetItem(t_item, [f"↳ {sub['text']}", ""])
                    s_item.setData(0, QtCore.Qt.UserRole, t["id"])
                    s_item.setData(1, QtCore.Qt.UserRole, idx)
                    s_item.setFlags(s_item.flags() | QtCore.Qt.ItemIsUserCheckable)
                    s_item.setCheckState(0, QtCore.Qt.Checked if sub.get("done") else QtCore.Qt.Unchecked)
                    if sub.get("done"):
                        s_item.setForeground(0, QtGui.QColor("#64748B"))
                t_item.setExpanded(True)

        self.tree.blockSignals(False)