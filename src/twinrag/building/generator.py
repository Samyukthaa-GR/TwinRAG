"""
Generate a building water-supply network from a ``BuildingSpec``.

``BuildingNetworkGenerator`` produces two artefacts from one design:

``WaterNetworkModel``
    An EPANET-ready hydraulic model. Every downstream stage -- the
    simulator, fault injectors, graph builder and detectors -- consumes
    this exactly as it consumes Net3, so the building needs no special
    handling anywhere in the pipeline.

``layout``
    A JSON-friendly dict carrying what EPANET cannot store: true 3D
    positions, the routed path of every pipe (runs below the slab, then
    drops to the fixture), and the building semantics -- which floor,
    flat and room every asset belongs to, and where the instruments are.
    The 3D viewer draws from it, and later phases use it to turn an
    asset ID into "kitchen of flat 3A".

Asset IDs are human-readable and never collide between nodes and links
(unlike Net3, where junction ``101`` and pipe ``101`` coexist):

    F3A-KIT        kitchen plumbing point, floor 3, flat A   (junction)
    F3A-KIT-BR     branch pipe feeding it                    (pipe)
    F3A-IN         flat inlet, after isolation valve + meter (junction)
    F3A-SUPPLY     flat supply pipe from the down-take       (pipe)
    DTA-3          down-take riser tap-off, stack A, floor 3 (junction)
    DTA-4-3        riser segment between floors 4 and 3      (pipe)
    OHT / SUMP     roof tank / underground sump              (tanks)
    PUMP1          transfer pump, sump -> roof tank          (pump)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import wntr
from wntr.network.base import LinkStatus
from wntr.network.controls import ControlAction, Rule, ValueCondition

from .spec import BuildingSpec


LAYOUT_SCHEMA = "twinrag.building_layout/1"

#: Oblique (cabinet) projection used for the 2D coordinates written into
#: the .inp file, so 2D tools such as the existing flat viewer or EPANET's
#: own GUI still show a recognisable building rather than seven floors
#: stacked on top of each other in plan.
_OBLIQUE_X = 0.45
_OBLIQUE_Y = 0.30


def _round_point(point) -> list:
    return [round(float(value), 3) for value in point]


def _path_length(path) -> float:
    return sum(
        math.dist(path[index], path[index + 1])
        for index in range(len(path) - 1)
    )


def _route(start, end, order: str = "xyz") -> list:
    """
    Orthogonal pipe route from ``start`` to ``end``.

    Real pipework runs along walls and below slabs, not diagonally
    through rooms, so a route moves one axis at a time in ``order``.
    Consecutive duplicate points are dropped.
    """

    axes = {"x": 0, "y": 1, "z": 2}
    current = list(start)
    points = [tuple(current)]

    for axis in order:
        index = axes[axis]

        if current[index] != end[index]:
            current[index] = end[index]
            points.append(tuple(current))

    return [_round_point(point) for point in points]


class BuildingNetworkGenerator:
    """
    Lay out and size a building's cold-water supply system.
    """

    def __init__(self, spec: BuildingSpec | None = None):
        self.spec = spec or BuildingSpec()
        self._validate_spec()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_spec(self) -> None:
        spec = self.spec

        if spec.floors < 1:
            raise ValueError("A building needs at least one floor.")

        if not 1 <= len(spec.flat_letters) <= 2:
            raise ValueError(
                "The mirrored plan supports one or two flats per floor."
            )

        if not spec.wet_rooms:
            raise ValueError("The flat plan has no wet rooms.")

        share = sum(room.demand_share for room in spec.wet_rooms)

        if not math.isclose(share, 1.0, abs_tol=1e-9):
            raise ValueError(
                f"Wet-room demand shares must sum to 1.0, got {share}."
            )

        for room in spec.wet_rooms:
            if room.tee not in ("K", "B"):
                raise ValueError(
                    f"Wet room '{room.code}' must be fed from tee "
                    f"'K' or 'B', got {room.tee!r}."
                )

            if room.pattern not in spec.patterns:
                raise ValueError(
                    f"Wet room '{room.code}' uses unknown demand "
                    f"pattern {room.pattern!r}."
                )

        for name, values in spec.patterns.items():
            if len(values) != 24 or min(values) < 0 or sum(values) <= 0:
                raise ValueError(
                    f"Pattern '{name}' must be 24 non-negative hourly "
                    "multipliers with a positive total."
                )

        if not (
            spec.roof_tank.min_level
            < spec.pump_start_level
            < spec.pump_stop_level
            < spec.roof_tank.max_level
        ):
            raise ValueError(
                "Pump level switches must satisfy min_level < start "
                "< stop < max_level."
            )

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _floor_z(self, floor: int) -> float:
        return floor * self.spec.floor_height

    def _plan_x(self, letter: str, local_x: float) -> float:
        """
        Map a flat-local x onto the building. Flat B is mirrored so its
        wet rooms also back onto the central core.
        """

        if letter == self.spec.flat_letters[0]:
            return local_x

        return self.spec.building_width - local_x

    def _plan_rect(self, letter: str, rect) -> list:
        x0, y0, x1, y1 = rect
        a = self._plan_x(letter, x0)
        b = self._plan_x(letter, x1)
        return [round(min(a, b), 3), y0, round(max(a, b), 3), y1]

    def _shaft_x(self, letter: str) -> float:
        return self._plan_x(
            letter,
            self.spec.flat_width + self.spec.shaft_offset,
        )

    @staticmethod
    def _floor_label(floor: int) -> str:
        return "Ground floor" if floor == 0 else f"Floor {floor}"

    @staticmethod
    def _oblique(point) -> tuple:
        x, y, z = point
        return (
            round(x + _OBLIQUE_X * y, 3),
            round(z + _OBLIQUE_Y * y, 3),
        )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self):
        """
        Returns
        -------
        (WaterNetworkModel, dict)
            The hydraulic model and its 3D layout.
        """

        spec = self.spec

        wn = wntr.network.WaterNetworkModel()
        self._configure_options(wn)
        self._add_patterns(wn)

        layout = {
            "schema": LAYOUT_SCHEMA,
            "building": self._building_summary(),
            "floors": [
                {
                    "index": floor,
                    "label": self._floor_label(floor),
                    "z": self._floor_z(floor),
                }
                for floor in range(spec.floors)
            ],
            "flats": [],
            "rooms": [],
            "nodes": {},
            "links": {},
            "sensors": [],
        }

        self._wn = wn
        self._layout = layout

        self._add_supply_side()

        for floor in range(spec.floors):
            for letter in spec.flat_letters:
                self._add_flat(floor, letter)

        self._add_downtakes()
        self._add_pump_controls()
        self._add_sensors()

        del self._wn, self._layout

        return wn, layout

    def _configure_options(self, wn) -> None:
        spec = self.spec
        options = wn.options

        options.time.duration = 24 * 3600
        options.time.hydraulic_timestep = 900
        options.time.pattern_timestep = 3600
        options.time.report_timestep = 3600
        options.time.rule_timestep = 60

        options.hydraulic.headloss = "H-W"
        options.hydraulic.inpfile_units = "LPS"
        options.hydraulic.demand_model = "PDA"
        options.hydraulic.required_pressure = spec.required_pressure
        options.hydraulic.minimum_pressure = spec.minimum_pressure
        options.hydraulic.pressure_exponent = 0.5

    def _add_patterns(self, wn) -> None:
        for name, values in self.spec.patterns.items():
            mean = sum(values) / len(values)
            wn.add_pattern(name, [value / mean for value in values])

    def _building_summary(self) -> dict:
        spec = self.spec
        width = spec.building_width

        return {
            "name": spec.name,
            "floors": spec.floors,
            "flats_per_floor": len(spec.flat_letters),
            "floor_height": spec.floor_height,
            "terrace_level": spec.terrace_level,
            "footprint": [0.0, 0.0, width, spec.flat_depth],
            "core": [
                spec.flat_width,
                0.0,
                spec.flat_width + spec.core_width,
                spec.flat_depth,
            ],
            "persons_per_flat": spec.persons_per_flat,
            "litres_per_person_day": spec.litres_per_person_day,
            "supply_scheme": (
                "municipal main -> underground sump -> transfer pump -> "
                "roof tank -> gravity down-take per flat stack"
            ),
            "units": {
                "length": "m",
                "flowrate": "m3/s",
                "pressure": "m (head)",
            },
        }

    # ------------------------------------------------------------------
    # Element helpers -- each adds to both the model and the layout
    # ------------------------------------------------------------------

    def _node_entry(self, node_id, position, role, label, **extra) -> None:
        self._layout["nodes"][node_id] = {
            "x": round(position[0], 3),
            "y": round(position[1], 3),
            "z": round(position[2], 3),
            "role": role,
            "label": label,
            "floor": extra.pop("floor", None),
            "flat": extra.pop("flat", None),
            "room": extra.pop("room", None),
            **extra,
        }

    def _junction(
        self,
        node_id,
        position,
        role,
        label,
        base_demand=0.0,
        pattern=None,
        **semantics,
    ) -> None:
        self._wn.add_junction(
            node_id,
            base_demand=base_demand,
            demand_pattern=pattern,
            elevation=position[2],
            coordinates=self._oblique(position),
        )
        self._node_entry(node_id, position, role, label, **semantics)

    def _pipe(self, link_id, start, end, path, role, label, **semantics):
        spec = self.spec

        self._wn.add_pipe(
            link_id,
            start,
            end,
            length=max(round(_path_length(path), 3), 0.3),
            diameter=spec.diameters[role],
            roughness=spec.roughness,
            minor_loss=spec.minor_losses[role],
        )

        self._layout["links"][link_id] = {
            "start": start,
            "end": end,
            "kind": "pipe",
            "role": role,
            "label": label,
            "diameter": spec.diameters[role],
            "length": round(_path_length(path), 3),
            "path": path,
            "floor": semantics.get("floor"),
            "flat": semantics.get("flat"),
            "room": semantics.get("room"),
        }

    def _position(self, node_id) -> tuple:
        node = self._layout["nodes"][node_id]
        return (node["x"], node["y"], node["z"])

    # ------------------------------------------------------------------
    # Supply side: municipal main, sump, pump, roof tank
    # ------------------------------------------------------------------

    def _add_supply_side(self) -> None:
        spec = self.spec
        wn = self._wn

        centre = spec.building_width / 2
        terrace = spec.terrace_level
        roof_tank_floor = terrace + spec.staging_height

        # Municipal connection at the road, south-east of the building.
        city = (spec.building_width + 2.0, -6.0, 0.0)
        wn.add_reservoir(
            "CITY",
            base_head=spec.municipal_head,
            coordinates=self._oblique(city),
        )
        self._node_entry(
            "CITY", city, "municipal_supply", "Municipal water main",
        )

        # Underground sump, in front of the building.
        sump = spec.sump
        sump_position = (centre + 5.0, -3.0, sump.elevation)
        wn.add_tank(
            "SUMP",
            elevation=sump.elevation,
            init_level=sump.init_level,
            min_level=sump.min_level,
            max_level=sump.max_level,
            diameter=sump.diameter,
            coordinates=self._oblique(sump_position),
        )
        self._node_entry(
            "SUMP", sump_position, "sump", "Underground sump",
            diameter=sump.diameter,
            max_level=sump.max_level,
            min_level=sump.min_level,
        )

        # Roof (overhead) tank on a stand above the terrace.
        tank = spec.roof_tank
        roof_position = (centre, spec.shaft_y + 2.5, roof_tank_floor)
        wn.add_tank(
            "OHT",
            elevation=roof_tank_floor,
            init_level=tank.init_level,
            min_level=tank.min_level,
            max_level=tank.max_level,
            diameter=tank.diameter,
            coordinates=self._oblique(roof_position),
        )
        self._node_entry(
            "OHT", roof_position, "roof_tank", "Overhead (roof) tank",
            diameter=tank.diameter,
            max_level=tank.max_level,
            min_level=tank.min_level,
        )

        # Pump room beside the sump.
        pump_delivery = (centre + 3.0, -1.2, 0.3)
        self._junction(
            "PUMP-DEL", pump_delivery, "pump_delivery",
            "Pump delivery header",
        )

        # Roof manifold where the tank outlet splits to the stacks.
        manifold = (centre, spec.shaft_y, terrace + 0.4)
        self._junction(
            "ROOF-MAN", manifold, "roof_manifold", "Roof distribution manifold",
        )

        # --- Links ------------------------------------------------------

        buried = -0.8
        self._pipe(
            "CITY-INLET", "CITY", "SUMP",
            [
                _round_point(city),
                _round_point((city[0], city[1], buried)),
                _round_point((sump_position[0], city[1], buried)),
                _round_point((sump_position[0], sump_position[1], buried)),
            ],
            "municipal_inlet", "Municipal inlet to sump",
        )

        wn.add_curve(
            "PUMP1-CURVE",
            "HEAD",
            [(spec.pump_design_flow, spec.pump_design_head)],
        )
        wn.add_pump(
            "PUMP1",
            "SUMP",
            "PUMP-DEL",
            pump_type="HEAD",
            pump_parameter="PUMP1-CURVE",
        )
        self._layout["links"]["PUMP1"] = {
            "start": "SUMP",
            "end": "PUMP-DEL",
            "kind": "pump",
            "role": "pump",
            "label": "Transfer pump (sump to roof tank)",
            "diameter": spec.diameters["rising_main"],
            "length": None,
            "path": [
                _round_point((sump_position[0], sump_position[1], 0.3)),
                _round_point(pump_delivery),
            ],
            "floor": None,
            "flat": None,
            "room": None,
        }

        riser_x = centre + 0.6
        tank_top = roof_tank_floor + tank.max_level + 0.2
        self._pipe(
            "RISING-MAIN", "PUMP-DEL", "OHT",
            [
                _round_point(pump_delivery),
                _round_point((riser_x, pump_delivery[1], pump_delivery[2])),
                _round_point((riser_x, spec.shaft_y - 0.6, pump_delivery[2])),
                _round_point((riser_x, spec.shaft_y - 0.6, tank_top)),
                _round_point((riser_x, roof_position[1], tank_top)),
            ],
            "rising_main", "Rising main to roof tank",
        )

        self._pipe(
            "OHT-OUTLET", "OHT", "ROOF-MAN",
            _route(roof_position, manifold, "xyz"),
            "tank_outlet", "Roof tank outlet",
        )

    # ------------------------------------------------------------------
    # One flat: inlet, two distribution tees, one branch per wet room
    # ------------------------------------------------------------------

    def _add_flat(self, floor: int, letter: str) -> None:
        spec = self.spec
        flat_id = f"F{floor}{letter}"
        flat_label = f"Flat {floor}{letter}"
        floor_z = self._floor_z(floor)
        run_z = floor_z + spec.pipe_run_height
        fixture_z = floor_z + spec.fixture_height

        semantics = {"floor": floor, "flat": flat_id}

        self._layout["flats"].append(
            {
                "id": flat_id,
                "label": flat_label,
                "floor": floor,
                "letter": letter,
                "rect": self._plan_rect(
                    letter, (0.0, 0.0, spec.flat_width, spec.flat_depth)
                ),
                "z": floor_z,
            }
        )

        for room in spec.rooms:
            self._layout["rooms"].append(
                {
                    "id": f"{flat_id}-{room.code}",
                    "code": room.code,
                    "name": room.name,
                    "flat": flat_id,
                    "floor": floor,
                    "rect": self._plan_rect(letter, room.rect),
                    "z": floor_z,
                    "wet": room.wet,
                    "fixture_node": (
                        f"{flat_id}-{room.code}" if room.wet else None
                    ),
                }
            )

        # Flat inlet: just inside the flat, beside the service shaft.
        inlet_id = f"{flat_id}-IN"
        inlet = (
            self._plan_x(letter, spec.flat_width - 0.3),
            spec.shaft_y,
            run_z,
        )
        self._junction(
            inlet_id, inlet, "flat_inlet",
            f"{flat_label} inlet (valve + meter)", **semantics,
        )

        tees = {
            "K": (
                f"{flat_id}-TK",
                (self._plan_x(letter, 9.6), 4.2, run_z),
                "kitchen-side distribution tee",
            ),
            "B": (
                f"{flat_id}-TB",
                (self._plan_x(letter, 7.4), 8.0, run_z),
                "bathroom-side distribution tee",
            ),
        }

        for key, (tee_id, position, description) in tees.items():
            self._junction(
                tee_id, position, "flat_tee",
                f"{flat_label} {description}", **semantics,
            )
            self._pipe(
                f"{flat_id}-MAIN-{key}", inlet_id, tee_id,
                _route(inlet, position, "xyz"),
                "flat_main", f"{flat_label} {description} main",
                **semantics,
            )

        base_flow = spec.flat_daily_demand_m3 / 86400.0

        for room in spec.wet_rooms:
            node_id = f"{flat_id}-{room.code}"
            position = (
                self._plan_x(letter, room.fixture[0]),
                room.fixture[1],
                fixture_z,
            )
            room_semantics = {**semantics, "room": node_id}

            self._junction(
                node_id, position, "fixture",
                f"{flat_label} {room.name.lower()}",
                base_demand=base_flow * room.demand_share,
                pattern=room.pattern,
                room_name=room.name,
                **room_semantics,
            )

            tee_id, tee_position, _ = tees[room.tee]
            self._pipe(
                f"{node_id}-BR", tee_id, node_id,
                _route(tee_position, position, "xyz"),
                "room_branch", f"{flat_label} {room.name.lower()} branch",
                **room_semantics,
            )

    # ------------------------------------------------------------------
    # Down-take risers: roof manifold -> every floor of each flat stack
    # ------------------------------------------------------------------

    def _add_downtakes(self) -> None:
        spec = self.spec
        manifold = self._position("ROOF-MAN")
        top = spec.floors - 1

        for letter in spec.flat_letters:
            x = self._shaft_x(letter)
            stack = f"DT{letter}"

            for floor in range(top, -1, -1):
                node_id = f"{stack}-{floor}"
                position = (
                    x,
                    spec.shaft_y,
                    self._floor_z(floor) + spec.pipe_run_height,
                )
                self._junction(
                    node_id, position, "downtake_tap",
                    f"Down-take {letter}, {self._floor_label(floor).lower()} "
                    "tap-off",
                    floor=floor,
                )

                if floor == top:
                    self._pipe(
                        f"ROOF-{letter}", "ROOF-MAN", node_id,
                        _route(manifold, position, "xyz"),
                        "roof_main", f"Roof main to down-take {letter}",
                    )
                else:
                    above = f"{stack}-{floor + 1}"
                    self._pipe(
                        f"{stack}-{floor + 1}-{floor}", above, node_id,
                        _route(self._position(above), position, "xyz"),
                        "downtake",
                        f"Down-take {letter} riser, floor {floor + 1} "
                        f"to {floor}",
                        floor=floor,
                    )

                flat_id = f"F{floor}{letter}"
                inlet_id = f"{flat_id}-IN"
                self._pipe(
                    f"{flat_id}-SUPPLY", node_id, inlet_id,
                    _route(position, self._position(inlet_id), "xyz"),
                    "flat_supply", f"Flat {floor}{letter} supply",
                    floor=floor, flat=flat_id,
                )

    # ------------------------------------------------------------------
    # Controls and instrumentation
    # ------------------------------------------------------------------

    def _add_pump_controls(self) -> None:
        """
        Roof-tank level switches start and stop the transfer pump.

        Written as EPANET *rules* at low priority rather than simple
        controls, so a timed pump outage (priority 6) always wins while
        it is active, and level control resumes as soon as it ends.
        """

        spec = self.spec
        wn = self._wn
        tank = wn.get_node("OHT")
        pump = wn.get_link("PUMP1")

        for name, relation, level, status in (
            ("PUMP1-START", "<", spec.pump_start_level, LinkStatus.Open),
            ("PUMP1-STOP", ">", spec.pump_stop_level, LinkStatus.Closed),
        ):
            rule = Rule(
                ValueCondition(tank, "level", relation, level),
                ControlAction(pump, "status", status),
                priority=3,
                name=name,
            )
            wn.add_control(name, rule)

    def _add_sensors(self) -> None:
        """
        Instrument the building the way it would be metered in practice.

        Room-level diagnosis needs a flow meter on every room branch:
        inside a flat the pipes are a few metres long, so a leak in the
        kitchen and one in a bathroom differ by millimetres of pressure
        -- far below a transducer's resolution -- but each moves its own
        branch's flow unmistakably.
        """

        sensors = []

        def add(sensor_id, asset_id, asset_type, parameter, label):
            sensors.append(
                {
                    "id": sensor_id,
                    "asset_id": asset_id,
                    "asset_type": asset_type,
                    "parameter": parameter,
                    "label": label,
                }
            )

        for link_id, link in self._layout["links"].items():
            if link["role"] in (
                "room_branch", "flat_supply", "pump", "tank_outlet",
                "municipal_inlet",
            ):
                add(
                    f"FM-{link_id}", link_id, "link", "flowrate",
                    f"Flow meter: {link['label']}",
                )

        for node_id, node in self._layout["nodes"].items():
            if node["role"] in ("flat_inlet", "roof_tank", "sump"):
                add(
                    f"PT-{node_id}", node_id, "node", "pressure",
                    (
                        f"Level sensor: {node['label']}"
                        if node["role"] in ("roof_tank", "sump")
                        else f"Pressure sensor: {node['label']}"
                    ),
                )

        self._layout["sensors"] = sensors

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def write(self, inp_path, layout_path):
        """
        Build the network and save it as ``.inp`` plus layout JSON.
        """

        wn, layout = self.build()

        inp_path = Path(inp_path)
        layout_path = Path(layout_path)
        inp_path.parent.mkdir(parents=True, exist_ok=True)
        layout_path.parent.mkdir(parents=True, exist_ok=True)

        layout["building"]["network_file"] = inp_path.name

        wntr.network.write_inpfile(wn, str(inp_path), units="LPS")

        with layout_path.open("w", encoding="utf-8") as handle:
            json.dump(layout, handle, indent=1)

        return wn, layout


def load_layout(path) -> dict:
    """
    Read a layout JSON written by ``BuildingNetworkGenerator.write``.
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        layout = json.load(handle)

    if layout.get("schema") != LAYOUT_SCHEMA:
        raise ValueError(
            f"{path} is not a building layout "
            f"(schema {layout.get('schema')!r})."
        )

    return layout
