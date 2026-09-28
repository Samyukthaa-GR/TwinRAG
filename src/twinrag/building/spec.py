"""
Design specification for a residential building water-supply system.

Everything the generator needs to lay out a real, constructible plumbing
system lives here as plain data: storey height, the flat floor plan, the
storage tanks, the transfer pump, pipe sizes and per-room water demand.

The default describes a ground-plus-six (G+6) residential block with two
flats per floor, supplied the way most Indian multi-storey buildings are:

    municipal main -> underground sump -> transfer pump
        -> rising main -> overhead (roof) tank
        -> gravity down-take riser per flat stack
        -> flat inlet (isolation valve + meter)
        -> flat distribution tees -> room branches -> fixtures

Units are SI throughout, matching WNTR's internal units: metres, cubic
metres per second, seconds. Plan coordinates are metres from the
building's south-west corner: ``x`` runs east along the frontage, ``y``
runs north into the plot, ``z`` is height above finished ground level.

Design values follow common building-services practice rather than one
specific code clause: 135 litres per person per day for residential use
with full flushing, gravity supply from a staged roof tank, pipe
velocities well under 2 m/s at peak hourly demand, and at least ~5 m of
residual head at the top-floor fixtures under normal operation.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RoomSpec:
    """
    One room of the flat floor plan.

    ``rect`` is ``(x0, y0, x1, y1)`` in the flat's local plan, for
    flat A (the western flat). Flat B is the mirror image. Rooms with a
    ``fixture`` are wet rooms: they get a plumbing point, a branch pipe
    and a share of the flat's daily demand.
    """

    code: str
    name: str
    rect: tuple
    fixture: tuple | None = None
    demand_share: float = 0.0
    pattern: str | None = None
    tee: str | None = None

    @property
    def wet(self) -> bool:
        return self.fixture is not None


#: Flat floor plan, 11 m x 12 m, for the western flat (A). The eastern
#: flat (B) is mirrored about the building's centre line so both flats'
#: wet rooms sit against the central service core.
#:
#: ``tee`` names which distribution tee feeds a wet room: ``K`` is the
#: kitchen-side tee (kitchen, utility), ``B`` is the bathroom-side tee.
DEFAULT_ROOMS: tuple = (
    RoomSpec("LIV", "Living / dining", (0.0, 0.0, 6.0, 6.5)),
    RoomSpec(
        "KIT", "Kitchen", (6.0, 0.0, 11.0, 3.8),
        fixture=(9.6, 1.0), demand_share=0.20, pattern="kitchen", tee="K",
    ),
    RoomSpec("PAS", "Passage", (6.0, 3.8, 8.6, 6.5)),
    RoomSpec(
        "UTL", "Utility / wash area", (8.6, 3.8, 11.0, 6.5),
        fixture=(10.3, 5.6), demand_share=0.15, pattern="utility", tee="K",
    ),
    RoomSpec("BED1", "Master bedroom", (0.0, 6.5, 3.8, 12.0)),
    RoomSpec(
        "MBATH", "Master bathroom", (3.8, 9.4, 5.8, 12.0),
        fixture=(4.8, 11.2), demand_share=0.25, pattern="bath", tee="B",
    ),
    RoomSpec(
        "CBATH", "Common bathroom", (3.8, 6.5, 5.8, 9.4),
        fixture=(4.8, 7.6), demand_share=0.20, pattern="bath", tee="B",
    ),
    RoomSpec("BED2", "Bedroom 2", (5.8, 6.5, 9.0, 12.0)),
    RoomSpec(
        "BATH2", "Bathroom 2", (9.0, 8.6, 11.0, 12.0),
        fixture=(10.0, 11.0), demand_share=0.20, pattern="bath", tee="B",
    ),
    RoomSpec("STR", "Store", (9.0, 6.5, 11.0, 8.6)),
)


#: Hourly demand multipliers, hour 0 to hour 23. Normalised to a mean of
#: 1.0 by the generator, so they shape *when* water is drawn without
#: changing *how much* is drawn per day.
#:
#: Bathrooms peak before work and school, kitchens around meal times,
#: and the utility area (washing machine, floor washing) mid-morning.
DEFAULT_PATTERNS: dict = {
    "bath": (
        0.2, 0.1, 0.1, 0.1, 0.3, 1.2, 3.0, 3.4, 2.6, 1.4, 0.8, 0.6,
        0.5, 0.5, 0.4, 0.5, 0.7, 1.0, 1.4, 1.6, 1.4, 1.1, 0.8, 0.4,
    ),
    "kitchen": (
        0.1, 0.1, 0.1, 0.1, 0.2, 0.6, 1.6, 2.2, 2.0, 1.2, 1.0, 1.6,
        2.0, 1.4, 0.8, 0.6, 0.8, 1.2, 1.8, 2.4, 2.2, 1.2, 0.5, 0.2,
    ),
    "utility": (
        0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.8, 1.6, 2.6, 3.0, 2.6, 1.8,
        1.2, 0.9, 0.8, 0.8, 0.9, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2,
    ),
}


#: Internal pipe diameters in metres, by pipe role.
DEFAULT_DIAMETERS: dict = {
    "municipal_inlet": 0.020,
    "rising_main": 0.050,
    "tank_outlet": 0.050,
    "roof_main": 0.040,
    "downtake": 0.040,
    "flat_supply": 0.025,
    "flat_main": 0.020,
    "room_branch": 0.015,
}


#: Minor-loss coefficients by pipe role: isolation valves, meters,
#: elbows and tees that a straight-pipe length does not capture.
DEFAULT_MINOR_LOSSES: dict = {
    "municipal_inlet": 12.0,
    "rising_main": 2.0,
    "tank_outlet": 1.5,
    "roof_main": 1.0,
    "downtake": 0.5,
    "flat_supply": 3.0,
    "flat_main": 1.0,
    "room_branch": 1.5,
}


@dataclass(frozen=True)
class TankSpec:
    """
    A cylindrical storage tank. ``elevation`` is the tank floor.
    """

    elevation: float
    diameter: float
    init_level: float
    min_level: float
    max_level: float


@dataclass(frozen=True)
class BuildingSpec:
    """
    Complete design of one building's cold-water supply system.
    """

    name: str = "TwinRAG Residency (G+6)"

    # --- Massing ------------------------------------------------------
    floors: int = 7                      # ground + 6 upper floors
    flat_letters: tuple = ("A", "B")     # two flats per floor
    floor_height: float = 3.0            # slab to slab
    flat_width: float = 11.0             # x extent of one flat
    flat_depth: float = 12.0             # y extent of one flat
    core_width: float = 4.0              # stair, lift and service shafts
    pipe_run_height: float = 2.7         # branch pipes run below the slab
    fixture_height: float = 0.9          # tap / cistern inlet height

    # --- Service shaft (per flat, against the core) ------------------
    shaft_offset: float = 0.4            # into the core from the flat edge
    shaft_y: float = 7.5

    # --- Occupancy and demand ----------------------------------------
    persons_per_flat: float = 4.5
    litres_per_person_day: float = 135.0

    rooms: tuple = DEFAULT_ROOMS
    patterns: dict = field(default_factory=lambda: dict(DEFAULT_PATTERNS))

    # --- Supply ------------------------------------------------------
    municipal_head: float = 7.0          # metres above ground at the road

    sump: TankSpec = TankSpec(
        elevation=-3.0,
        diameter=3.0,
        init_level=2.0,
        min_level=0.2,
        max_level=2.5,
    )

    staging_height: float = 3.0          # roof tank stand above terrace

    roof_tank: TankSpec = TankSpec(
        elevation=0.0,                   # set from terrace + staging
        diameter=2.0,
        init_level=1.4,
        min_level=0.15,
        max_level=2.0,
    )

    # Single-point pump curve: design flow (m3/s) at design head (m).
    pump_design_flow: float = 0.002
    pump_design_head: float = 38.0

    # Level switches in the roof tank that start and stop the pump.
    pump_start_level: float = 0.6
    pump_stop_level: float = 1.9

    # --- Hydraulics --------------------------------------------------
    roughness: float = 140.0             # Hazen-Williams C, CPVC / uPVC
    diameters: dict = field(default_factory=lambda: dict(DEFAULT_DIAMETERS))
    minor_losses: dict = field(
        default_factory=lambda: dict(DEFAULT_MINOR_LOSSES)
    )

    # Pressure-driven demand: a tap delivers its full flow at or above
    # ``required_pressure`` and nothing at ``minimum_pressure``. A tree
    # network with no redundancy needs this -- close one riser under a
    # demand-driven model and EPANET cannot balance the floors below it.
    required_pressure: float = 3.0
    minimum_pressure: float = 0.0

    @property
    def terrace_level(self) -> float:
        return self.floors * self.floor_height

    @property
    def building_width(self) -> float:
        return 2 * self.flat_width + self.core_width

    @property
    def flat_daily_demand_m3(self) -> float:
        return self.persons_per_flat * self.litres_per_person_day / 1000.0

    @property
    def wet_rooms(self) -> tuple:
        return tuple(room for room in self.rooms if room.wet)
