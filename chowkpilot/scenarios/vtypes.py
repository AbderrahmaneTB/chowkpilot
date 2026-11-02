"""Indian mixed-traffic vehicle types for SUMO's sublane model.

Drivers merge assertively at unsignalised junctions (impatience, short minor-road
time gaps, nosing into slow foe traffic), as Indian traffic does; SUMO defaults
would leave side-street traffic waiting indefinitely for gaps.

Two-wheelers and autos use `latAlignment="arbitrary"` with a small lateral gap, so
they filter between cars and queue side by side at the stop line, as they do on
Indian roads. Cars, buses and trucks stay lane-centred.

Default traffic mix (share of vehicles entering the network). Telangana's
registered fleet on 31 Mar 2025 was 73.4% motorcycles, 13.5% cars and 3.0% autos
(transport.telangana.gov.in); on-road shares at urban arterials carry more cars,
autos and buses than the registered fleet, so the mix below is shifted toward them.
"""

from xml.sax.saxutils import quoteattr

VEHICLE_TYPES = {
    "two_wheeler": dict(
        impatience=0.6, jmTimegapMinor=0.5, jmIgnoreFoeProb=0.2, jmIgnoreFoeSpeed=4.0,
        vClass="motorcycle", length=2.0, width=0.8, minGap=0.6, accel=2.8, decel=5.0,
        emergencyDecel=8.0, maxSpeed=16.67, sigma=0.6, speedFactor="normc(1.0,0.15,0.6,1.4)",
        latAlignment="arbitrary", minGapLat=0.25, maxSpeedLat=1.2,
        lcSublane=4.0, lcPushy=0.6, lcAssertive=1.5, color="0.98,0.75,0.18",
        emissionClass="HBEFA4/MC_4S_le250cc_Euro-3",
    ),
    "car": dict(
        impatience=0.4, jmTimegapMinor=0.8, jmIgnoreFoeProb=0.1, jmIgnoreFoeSpeed=2.0,
        vClass="passenger", length=4.2, width=1.75, minGap=1.5, accel=2.6, decel=4.5,
        emergencyDecel=8.0, maxSpeed=16.67, sigma=0.5, speedFactor="normc(1.0,0.1,0.7,1.3)",
        latAlignment="center", minGapLat=0.5, lcSublane=1.0, lcPushy=0.3,
        color="0.36,0.62,0.98", emissionClass="HBEFA4/PC_petrol_Euro-4",
    ),
    "auto_rickshaw": dict(
        impatience=0.5, jmTimegapMinor=0.6, jmIgnoreFoeProb=0.15, jmIgnoreFoeSpeed=3.0,
        vClass="passenger", length=2.8, width=1.4, minGap=1.0, accel=1.8, decel=4.0,
        emergencyDecel=7.0, maxSpeed=13.0, sigma=0.6, speedFactor="normc(1.0,0.1,0.7,1.2)",
        latAlignment="arbitrary", minGapLat=0.35, maxSpeedLat=1.0, lcSublane=2.0,
        lcPushy=0.5, color="0.2,0.78,0.42", emissionClass="HBEFA4/PC_petrol_Euro-3",
    ),
    "bus": dict(
        impatience=0.3, jmTimegapMinor=1.0,
        vClass="bus", length=11.0, width=2.5, minGap=2.0, accel=1.2, decel=4.0,
        emergencyDecel=7.0, maxSpeed=13.9, sigma=0.5, latAlignment="center", minGapLat=0.6,
        color="0.93,0.33,0.31", emissionClass="HBEFA4/UBus_Std_gt15-18t_Euro-IV_SCR",
    ),
    "lcv": dict(
        impatience=0.3, jmTimegapMinor=1.0,
        vClass="delivery", length=6.0, width=2.1, minGap=1.8, accel=1.5, decel=4.0,
        emergencyDecel=7.0, maxSpeed=13.9, sigma=0.5, latAlignment="center", minGapLat=0.5,
        color="0.66,0.55,0.95", emissionClass="HBEFA4/LCV_diesel_N1-III_Euro-4",
    ),
    "ambulance": dict(
        impatience=1.0, jmTimegapMinor=0.5,
        vClass="emergency", length=5.2, width=2.0, minGap=1.5, accel=2.5, decel=4.5,
        emergencyDecel=8.0, maxSpeed=19.4, speedFactor=1.15, sigma=0.3,
        latAlignment="center", minGapLat=0.5, color="1,1,1",
        emissionClass="HBEFA4/LCV_diesel_N1-III_Euro-4",
    ),
}

INDIAN_MIX = {"two_wheeler": 0.52, "car": 0.27, "auto_rickshaw": 0.12, "bus": 0.04, "lcv": 0.05}

# Order used by the replay format and the web renderer.
TYPE_INDEX = {name: i for i, name in enumerate(VEHICLE_TYPES)}

# Passenger-car units (IRC:106-1990 urban values) used by the Webster baseline.
PCU = {"two_wheeler": 0.5, "car": 1.0, "auto_rickshaw": 1.2, "bus": 2.2, "lcv": 1.4, "ambulance": 1.0}


def vtypes_xml(mix: dict | None = None, emission_classes: bool = True) -> str:
    """XML fragment defining all vehicle types plus the 'mix' distribution."""
    mix = mix or INDIAN_MIX
    lines = []
    for name, attrs in VEHICLE_TYPES.items():
        a = dict(attrs)
        if not emission_classes:
            a.pop("emissionClass", None)
        body = " ".join(f"{k}={quoteattr(str(v))}" for k, v in a.items())
        lines.append(f'    <vType id="{name}" {body}/>')
    members = " ".join(mix)
    probs = " ".join(f"{p:.4f}" for p in mix.values())
    lines.append(f'    <vTypeDistribution id="mix" vTypes="{members}" probabilities="{probs}"/>')
    return "\n".join(lines)
