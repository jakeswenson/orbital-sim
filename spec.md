# Orbital Dynamics Simulator — Project Specification

## Overview

An interactive web-based educational tool for understanding orbital mechanics. The primary goal is to give users visceral, intuitive understanding of how orbital motion works by letting them control a spacecraft and see the (often counterintuitive) results of their actions.

**Target audience:** People who don't yet understand orbital mechanics, specifically addressing common misconceptions.

**Tech stack:** Svelte 5 + TypeScript, Canvas 2D rendering, pure functional physics engine.

---

## Educational Objectives

### Misconceptions to Address

#### 1. "Point at target and thrust"

**The misconception:** To go somewhere (like the Moon), you point your spacecraft at it and accelerate.

**The reality:** Your spacecraft is already moving at ~7.7 km/s tangent to your orbit. Thrusting toward the Moon mostly changes your orbital shape, not your direction. To reach the Moon, you thrust *prograde* (in the direction you're already moving) to raise your orbit until it intersects the Moon's orbit.

**How the sim teaches this:** Real-time trajectory prediction line that updates as user rotates the spacecraft. User sees that pointing at the Moon doesn't create a path to the Moon.

#### 2. "The Moon is impossibly far"

**The misconception:** The ISS is 250 miles up, the Moon is 238,000 miles away—that's ~1000x farther, which seems impossibly far for 1960s technology.

**The reality:** The ISS orbits at 7.66 km/s, completing an orbit in 90 minutes. At these speeds, a 3-day journey to the Moon is entirely reasonable. The trans-lunar injection burn only adds ~3.1 km/s to your velocity.

**How the sim teaches this:** 
- Display actual velocities in the HUD
- Show time-to-destination estimates
- Time acceleration controls let user experience the journey at compressed time

#### 3. "Reverse thrust to orbit backwards"

**The misconception:** To orbit in the opposite direction, just rotate 180° and thrust while "resisting" gravity.

**The reality:** Your orbital velocity IS what's keeping you from falling. Thrusting retrograde immediately starts lowering your orbit. To actually reverse your orbit, you'd need to:
1. Kill 100% of your orbital velocity (you'd start falling straight down)
2. Accelerate to full orbital velocity in the opposite direction
3. Total Δv cost: 2× orbital velocity (for ISS: 15.3 km/s—more than going from Earth's surface to orbit)

**How the sim teaches this:**
- When user thrusts retrograde, they see periapsis dropping
- "Δv to circular orbit" display shows the growing "orbital debt"
- Trajectory prediction shows the orbit collapsing toward Earth
- If they kill too much velocity, trajectory prediction shows impact

---

## Technical Decisions

### Physics Model

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Physics model | Newtonian N-body | GR effects at Earth-Moon distances are ~10⁻⁹—imperceptible. True N-body over patched conics for honest physics. |
| Integrator | Velocity Verlet | Better energy conservation than RK4 for long orbital simulations. Symplectic integrator preserves orbital characteristics. |
| Bodies | Sun, Earth, Moon | Minimal set to demonstrate Earth orbit and lunar transfer. Mars deferred to v2. |
| Coordinate system | Heliocentric inertial | Sun at origin. Simplifies N-body calculations. |

### Rendering

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Dimensions | 2D (top-down ecliptic view) | Clearer for teaching. 3D adds complexity without educational value for these misconceptions. |
| Renderer | Canvas 2D | Simple, performant, sufficient for 2D. No library dependencies. |
| Scale handling | Zoom + optional compression | True scale is important for understanding, but need usability. Objects get minimum visual size at extreme zoom-out. |

### Framework

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Framework | Svelte 5 | User preference. Runes ($state, $derived) are clean for reactive simulation state. |
| Language | TypeScript | Required. Strict mode. |
| State management | Svelte 5 runes | No external state library needed. |
| Physics execution | Main thread (initially) | Web Worker if performance issues arise. |

---

## Architecture

```
src/
├── lib/
│   ├── physics/           # Pure functions, no framework dependencies
│   │   ├── types.ts       # Core type definitions
│   │   ├── constants.ts   # G, masses, radii, orbital parameters
│   │   ├── vector.ts      # Vec2 operations (pure functions)
│   │   ├── gravity.ts     # Gravitational acceleration calculations
│   │   ├── integrator.ts  # Velocity Verlet step function
│   │   ├── orbits.ts      # Orbital element calculations
│   │   └── prediction.ts  # Trajectory propagation
│   │
│   ├── simulation/        # Svelte-aware state and control
│   │   ├── state.svelte.ts    # Reactive simulation state (runes)
│   │   ├── controls.ts        # Input handling
│   │   └── loop.ts            # requestAnimationFrame loop
│   │
│   ├── rendering/         # Canvas rendering functions
│   │   ├── context.ts     # Canvas setup and coordinate transforms
│   │   ├── bodies.ts      # Planet/moon rendering
│   │   ├── spacecraft.ts  # Ship + thrust indicator
│   │   ├── trajectory.ts  # Predicted path + history trail
│   │   ├── orbits.ts      # Orbital path rendering
│   │   ├── gravity-field.ts   # Gravity well visualization
│   │   └── hud.ts         # On-canvas HUD elements (optional)
│   │
│   └── components/        # Svelte components
│       ├── Simulation.svelte  # Main canvas + game loop
│       ├── HUD.svelte         # Velocity/orbital data overlay
│       ├── Controls.svelte    # Time warp, zoom, settings
│       └── Tutorial.svelte    # Guided scenarios (v2)
│
├── routes/
│   └── +page.svelte       # Main app entry
│
├── app.html
├── app.css
└── tests/
    └── physics/           # Unit tests for physics engine
        ├── vector.test.ts
        ├── gravity.test.ts
        └── integrator.test.ts
```

### Design Principles

1. **Physics engine is pure functions** — No side effects, no framework dependencies. Fully testable. Could be extracted to separate package.

2. **Immutable state** — SimState is readonly. Each physics step returns a new state. Enables time-travel debugging, trajectory prediction.

3. **Separation of concerns** — Physics knows nothing about rendering. Rendering knows nothing about Svelte. Components orchestrate.

4. **Functional core, imperative shell** — Pure physics calculations wrapped by imperative game loop and event handlers.

---

## Core Types (Contracts)

```typescript
// =============================================================================
// lib/physics/types.ts
// =============================================================================

/** Immutable 2D vector */
export interface Vec2 {
  readonly x: number;
  readonly y: number;
}

/** Celestial body (planet, moon, sun) */
export interface Body {
  readonly id: string;
  readonly name: string;
  readonly mass: number;           // kg
  readonly radius: number;         // meters
  readonly position: Vec2;         // meters, heliocentric inertial frame
  readonly velocity: Vec2;         // m/s
  readonly color: string;          // CSS color for rendering
  readonly orbitColor?: string;    // CSS color for orbit path
}

/** Player-controlled spacecraft */
export interface Spacecraft {
  readonly position: Vec2;         // meters, heliocentric
  readonly velocity: Vec2;         // m/s
  readonly heading: number;        // radians, 0 = +x axis, counterclockwise
  readonly mass: number;           // kg (small, mostly cosmetic)
  readonly thrustAccel: number;    // m/s² when engine firing
}

/** Complete simulation state at a point in time */
export interface SimState {
  readonly time: number;           // seconds since epoch (J2000)
  readonly bodies: readonly Body[];
  readonly spacecraft: Spacecraft;
  readonly isThrusting: boolean;
  readonly rotationInput: -1 | 0 | 1;  // -1 = CCW, 0 = none, 1 = CW
}

/** View/camera state (not part of physics) */
export interface ViewState {
  readonly center: Vec2;           // world coords of view center
  readonly zoom: number;           // pixels per meter
  readonly followBody: string | 'spacecraft' | null;  // ID of body to track
  readonly showTrueScale: boolean;
  readonly showGravityField: boolean;
  readonly showTrajectory: boolean;
  readonly showOrbits: boolean;
  readonly showVelocityVectors: boolean;
  readonly timeScale: number;      // simulation seconds per real second
}

/** Keplerian orbital elements (computed from state vectors) */
export interface OrbitalElements {
  readonly semiMajorAxis: number;      // meters
  readonly eccentricity: number;       // 0 = circular, <1 = ellipse, >=1 = escape
  readonly apoapsis: number;           // meters from body center (NaN if escape)
  readonly periapsis: number;          // meters from body center
  readonly apoapsisAltitude: number;   // meters above surface
  readonly periapsisAltitude: number;  // meters above surface
  readonly period: number;             // seconds (NaN if escape)
  readonly trueAnomaly: number;        // radians
  readonly argumentOfPeriapsis: number; // radians
  readonly specificEnergy: number;     // J/kg (negative = bound, positive = escape)
  readonly isEscape: boolean;          // true if on escape trajectory
}

/** Point along a predicted trajectory */
export interface TrajectoryPoint {
  readonly position: Vec2;
  readonly velocity: Vec2;
  readonly time: number;
}

/** Relative state between spacecraft and a body */
export interface RelativeState {
  readonly body: Body;
  readonly distance: number;           // meters
  readonly relativeVelocity: Vec2;     // m/s
  readonly closingSpeed: number;       // m/s (positive = approaching)
  readonly orbitalElements: OrbitalElements;
}

/** HUD display data (derived from SimState) */
export interface HUDData {
  readonly spacecraftVelocity: number;     // m/s magnitude
  readonly spacecraftHeading: number;      // degrees for display
  readonly isThrusting: boolean;
  readonly relativeStates: RelativeState[];
  readonly primaryBody: RelativeState;     // body with strongest gravitational influence
  readonly deltaVToCircularize: number;    // m/s to make current orbit circular
  readonly deltaVToMoonTransfer: number;   // m/s for Hohmann to Moon
  readonly timeToImpact: number | null;    // seconds, if trajectory intersects body
}
```

---

## Physics Constants

```typescript
// =============================================================================
// lib/physics/constants.ts
// =============================================================================

/** Gravitational constant */
export const G = 6.67430e-11;  // m³/(kg·s²)

/** Astronomical unit (not used much for Earth-Moon, but useful) */
export const AU = 1.496e11;  // meters

/** Body data (approximate, epoch J2000) */
export const SUN = {
  id: 'sun',
  name: 'Sun',
  mass: 1.989e30,
  radius: 6.957e8,
  color: '#FDB813',
} as const;

export const EARTH = {
  id: 'earth',
  name: 'Earth',
  mass: 5.972e24,
  radius: 6.371e6,
  color: '#6B93D6',
  orbitColor: '#4A6FA5',
  semiMajorAxis: 1.496e11,     // 1 AU
  orbitalPeriod: 365.25 * 24 * 60 * 60,  // seconds
} as const;

export const MOON = {
  id: 'moon',
  name: 'Moon',
  mass: 7.342e22,
  radius: 1.737e6,
  color: '#C4C4C4',
  orbitColor: '#888888',
  semiMajorAxis: 3.844e8,      // from Earth
  orbitalPeriod: 27.3 * 24 * 60 * 60,  // seconds
} as const;

/** ISS orbital parameters (good starting point for spacecraft) */
export const ISS_ORBIT = {
  altitude: 420e3,             // 420 km above Earth surface
  velocity: 7660,              // m/s
  period: 92.68 * 60,          // seconds
} as const;

/** Spacecraft defaults */
export const SPACECRAFT_DEFAULTS = {
  mass: 1000,                  // kg
  thrustAccel: 10,             // m/s² (generous for gameplay)
  rotationRate: Math.PI / 2,   // rad/s (90°/s)
} as const;
```

---

## Vector Operations

```typescript
// =============================================================================
// lib/physics/vector.ts
// Pure functions for 2D vector math
// =============================================================================

import type { Vec2 } from './types';

export const vec2 = (x: number, y: number): Vec2 => ({ x, y });

export const ZERO: Vec2 = { x: 0, y: 0 };

export const add = (a: Vec2, b: Vec2): Vec2 => ({
  x: a.x + b.x,
  y: a.y + b.y,
});

export const sub = (a: Vec2, b: Vec2): Vec2 => ({
  x: a.x - b.x,
  y: a.y - b.y,
});

export const scale = (v: Vec2, s: number): Vec2 => ({
  x: v.x * s,
  y: v.y * s,
});

export const mag = (v: Vec2): number =>
  Math.sqrt(v.x * v.x + v.y * v.y);

export const magSq = (v: Vec2): number =>
  v.x * v.x + v.y * v.y;

export const norm = (v: Vec2): Vec2 => {
  const m = mag(v);
  return m === 0 ? ZERO : scale(v, 1 / m);
};

export const dot = (a: Vec2, b: Vec2): number =>
  a.x * b.x + a.y * b.y;

// 2D cross product (returns scalar z-component)
export const cross = (a: Vec2, b: Vec2): number =>
  a.x * b.y - a.y * b.x;

export const rotate = (v: Vec2, angle: number): Vec2 => {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return {
    x: v.x * cos - v.y * sin,
    y: v.x * sin + v.y * cos,
  };
};

export const angle = (v: Vec2): number =>
  Math.atan2(v.y, v.x);

export const fromAngle = (angle: number, magnitude: number = 1): Vec2 => ({
  x: Math.cos(angle) * magnitude,
  y: Math.sin(angle) * magnitude,
});

export const lerp = (a: Vec2, b: Vec2, t: number): Vec2 => ({
  x: a.x + (b.x - a.x) * t,
  y: a.y + (b.y - a.y) * t,
});

export const dist = (a: Vec2, b: Vec2): number =>
  mag(sub(b, a));

export const distSq = (a: Vec2, b: Vec2): number =>
  magSq(sub(b, a));
```

---

## Gravity Calculations

```typescript
// =============================================================================
// lib/physics/gravity.ts
// =============================================================================

import type { Vec2, Body } from './types';
import { sub, scale, mag, add, ZERO } from './vector';
import { G } from './constants';

/**
 * Gravitational acceleration on a point mass due to a single body.
 * Returns acceleration vector pointing toward the body.
 */
export const gravityFrom = (point: Vec2, body: Body): Vec2 => {
  const r = sub(body.position, point);
  const distSq = r.x * r.x + r.y * r.y;
  const dist = Math.sqrt(distSq);
  
  if (dist < body.radius) {
    // Inside the body - for simplicity, return zero or cap it
    // (in reality would need shell theorem)
    return ZERO;
  }
  
  const accelMag = (G * body.mass) / distSq;
  return scale(r, accelMag / dist);  // r/dist = unit vector toward body
};

/**
 * Total gravitational acceleration from all bodies.
 */
export const totalGravity = (point: Vec2, bodies: readonly Body[]): Vec2 =>
  bodies.reduce(
    (acc, body) => add(acc, gravityFrom(point, body)),
    ZERO
  );

/**
 * Gravitational potential at a point (for visualization).
 * More negative = deeper in gravity well.
 */
export const gravitationalPotential = (point: Vec2, bodies: readonly Body[]): number =>
  bodies.reduce((total, body) => {
    const dist = mag(sub(body.position, point));
    return total - (G * body.mass) / dist;
  }, 0);

/**
 * Find which body has the strongest gravitational influence at a point.
 * Returns the body and the acceleration magnitude.
 */
export const dominantBody = (
  point: Vec2, 
  bodies: readonly Body[]
): { body: Body; acceleration: number } => {
  let maxAccel = 0;
  let dominant = bodies[0];
  
  for (const body of bodies) {
    const r = sub(body.position, point);
    const distSq = r.x * r.x + r.y * r.y;
    const accel = (G * body.mass) / distSq;
    
    if (accel > maxAccel) {
      maxAccel = accel;
      dominant = body;
    }
  }
  
  return { body: dominant, acceleration: maxAccel };
};
```

---

## Integrator

```typescript
// =============================================================================
// lib/physics/integrator.ts
// Velocity Verlet integration for orbital mechanics
// =============================================================================

import type { SimState, Body, Spacecraft, Vec2 } from './types';
import { totalGravity } from './gravity';
import { add, scale, fromAngle } from './vector';
import { SPACECRAFT_DEFAULTS } from './constants';

/**
 * Compute acceleration on spacecraft given current state.
 */
const spacecraftAcceleration = (
  spacecraft: Spacecraft,
  bodies: readonly Body[],
  isThrusting: boolean
): Vec2 => {
  // Gravitational acceleration
  const gravity = totalGravity(spacecraft.position, bodies);
  
  // Thrust acceleration (if engine firing)
  if (isThrusting) {
    const thrust = fromAngle(spacecraft.heading, spacecraft.thrustAccel);
    return add(gravity, thrust);
  }
  
  return gravity;
};

/**
 * Velocity Verlet integration step.
 * 
 * Algorithm:
 *   x₁ = x₀ + v₀·dt + ½a₀·dt²
 *   a₁ = acceleration(x₁)
 *   v₁ = v₀ + ½(a₀ + a₁)·dt
 * 
 * This is a symplectic integrator - better energy conservation
 * than RK4 for orbital mechanics over long time spans.
 */
export const verletStep = (state: SimState, dt: number): SimState => {
  const { spacecraft, bodies, isThrusting, rotationInput } = state;
  
  // Update heading based on rotation input
  const newHeading = spacecraft.heading + 
    rotationInput * SPACECRAFT_DEFAULTS.rotationRate * dt;
  
  // Current acceleration
  const a0 = spacecraftAcceleration(spacecraft, bodies, isThrusting);
  
  // Position update: x₁ = x₀ + v₀·dt + ½a₀·dt²
  const newPos = add(
    add(spacecraft.position, scale(spacecraft.velocity, dt)),
    scale(a0, 0.5 * dt * dt)
  );
  
  // Create intermediate spacecraft state for acceleration calc
  const intermediateSpacecraft: Spacecraft = {
    ...spacecraft,
    position: newPos,
    heading: newHeading,
  };
  
  // New acceleration at new position
  const a1 = spacecraftAcceleration(intermediateSpacecraft, bodies, isThrusting);
  
  // Velocity update: v₁ = v₀ + ½(a₀ + a₁)·dt
  const newVel = add(
    spacecraft.velocity,
    scale(add(a0, a1), 0.5 * dt)
  );
  
  return {
    ...state,
    time: state.time + dt,
    spacecraft: {
      ...spacecraft,
      position: newPos,
      velocity: newVel,
      heading: newHeading,
    },
  };
};

/**
 * Update celestial body positions.
 * For v1, we use simple circular orbits (Keplerian).
 * Could upgrade to full N-body for bodies too.
 */
export const updateBodies = (bodies: readonly Body[], dt: number): Body[] => {
  // For now, assume bodies follow fixed circular orbits
  // This is simpler and sufficient for Earth-Moon demonstration
  // TODO: Full N-body for bodies if needed
  return bodies.map(body => {
    // Bodies maintain their velocities (circular orbit approximation)
    return {
      ...body,
      position: add(body.position, scale(body.velocity, dt)),
    };
  });
};

/**
 * Full simulation step: update spacecraft and bodies.
 */
export const simulationStep = (state: SimState, dt: number): SimState => {
  // Update spacecraft (including gravitational effects from all bodies)
  const stateAfterSpacecraft = verletStep(state, dt);
  
  // Update body positions
  const newBodies = updateBodies(stateAfterSpacecraft.bodies, dt);
  
  return {
    ...stateAfterSpacecraft,
    bodies: newBodies,
  };
};

/**
 * Adaptive time stepping for stability.
 * Subdivides large dt into smaller steps.
 */
export const adaptiveStep = (
  state: SimState, 
  dt: number, 
  maxStepSize: number = 60  // max 60 seconds per substep
): SimState => {
  const steps = Math.ceil(dt / maxStepSize);
  const subDt = dt / steps;
  
  let currentState = state;
  for (let i = 0; i < steps; i++) {
    currentState = simulationStep(currentState, subDt);
  }
  
  return currentState;
};
```

---

## Orbital Elements Calculation

```typescript
// =============================================================================
// lib/physics/orbits.ts
// Compute Keplerian orbital elements from state vectors
// =============================================================================

import type { Vec2, OrbitalElements, Body, Spacecraft } from './types';
import { sub, mag, magSq, dot, cross, scale, norm } from './vector';
import { G } from './constants';

/**
 * Compute orbital elements for a spacecraft relative to a central body.
 * Uses vis-viva equation and angular momentum.
 */
export const computeOrbitalElements = (
  spacecraft: Spacecraft,
  body: Body
): OrbitalElements => {
  // Position and velocity relative to body
  const r = sub(spacecraft.position, body.position);
  const v = sub(spacecraft.velocity, body.velocity);
  
  const rMag = mag(r);
  const vMag = mag(v);
  const mu = G * body.mass;  // Standard gravitational parameter
  
  // Specific orbital energy: ε = v²/2 - μ/r
  const specificEnergy = (vMag * vMag) / 2 - mu / rMag;
  
  // Specific angular momentum: h = r × v (scalar in 2D)
  const h = cross(r, v);
  
  // Semi-major axis: a = -μ/(2ε)
  // Negative energy = bound orbit, positive = escape
  const isEscape = specificEnergy >= 0;
  const semiMajorAxis = isEscape ? Infinity : -mu / (2 * specificEnergy);
  
  // Eccentricity vector: e = (v × h)/μ - r/|r|
  // In 2D: e = ((v² - μ/r)·r - (r·v)·v) / μ
  const eCrossH = { x: -v.y * h, y: v.x * h };  // v × h in 2D
  const eVec = sub(
    scale(eCrossH, 1 / mu),
    norm(r)
  );
  const eccentricity = mag(eVec);
  
  // Periapsis and apoapsis
  const periapsis = isEscape 
    ? semiMajorAxis * (1 - eccentricity)  // Still valid for hyperbola
    : semiMajorAxis * (1 - eccentricity);
  const apoapsis = isEscape 
    ? NaN 
    : semiMajorAxis * (1 + eccentricity);
  
  // Altitudes (above surface)
  const periapsisAltitude = periapsis - body.radius;
  const apoapsisAltitude = isEscape ? NaN : apoapsis - body.radius;
  
  // Orbital period: T = 2π√(a³/μ)
  const period = isEscape 
    ? NaN 
    : 2 * Math.PI * Math.sqrt(Math.pow(semiMajorAxis, 3) / mu);
  
  // True anomaly: angle from periapsis to current position
  const eNorm = eccentricity > 1e-10 ? norm(eVec) : { x: 1, y: 0 };
  const rNorm = norm(r);
  let trueAnomaly = Math.acos(Math.max(-1, Math.min(1, dot(eNorm, rNorm))));
  if (dot(r, v) < 0) {
    trueAnomaly = 2 * Math.PI - trueAnomaly;
  }
  
  // Argument of periapsis: angle from reference direction to periapsis
  const argumentOfPeriapsis = Math.atan2(eVec.y, eVec.x);
  
  return {
    semiMajorAxis,
    eccentricity,
    apoapsis,
    periapsis,
    apoapsisAltitude,
    periapsisAltitude,
    period,
    trueAnomaly,
    argumentOfPeriapsis,
    specificEnergy,
    isEscape,
  };
};

/**
 * Compute delta-v needed to circularize at current altitude.
 */
export const deltaVToCircularize = (
  spacecraft: Spacecraft,
  body: Body
): number => {
  const r = sub(spacecraft.position, body.position);
  const v = sub(spacecraft.velocity, body.velocity);
  const rMag = mag(r);
  const vMag = mag(v);
  const mu = G * body.mass;
  
  // Circular velocity at current radius
  const vCircular = Math.sqrt(mu / rMag);
  
  // Current velocity magnitude
  // Delta-v is difference (simplified - actual burn direction matters)
  return Math.abs(vCircular - vMag);
};

/**
 * Compute delta-v for Hohmann transfer to target orbit.
 * Returns total delta-v (two burns).
 */
export const deltaVHohmann = (
  currentRadius: number,
  targetRadius: number,
  mu: number
): { total: number; burn1: number; burn2: number } => {
  const r1 = currentRadius;
  const r2 = targetRadius;
  
  // Transfer orbit semi-major axis
  const a = (r1 + r2) / 2;
  
  // Velocities
  const v1 = Math.sqrt(mu / r1);  // Circular at r1
  const v2 = Math.sqrt(mu / r2);  // Circular at r2
  const vTransfer1 = Math.sqrt(mu * (2/r1 - 1/a));  // At periapsis of transfer
  const vTransfer2 = Math.sqrt(mu * (2/r2 - 1/a));  // At apoapsis of transfer
  
  const burn1 = Math.abs(vTransfer1 - v1);
  const burn2 = Math.abs(v2 - vTransfer2);
  
  return { total: burn1 + burn2, burn1, burn2 };
};
```

---

## Trajectory Prediction

```typescript
// =============================================================================
// lib/physics/prediction.ts
// Propagate trajectory forward for visualization
// =============================================================================

import type { SimState, TrajectoryPoint, Body } from './types';
import { simulationStep } from './integrator';
import { mag, sub } from './vector';

export interface PredictionResult {
  points: TrajectoryPoint[];
  impactBody: Body | null;  // If trajectory hits a body
  impactTime: number | null;
  escapes: boolean;          // If trajectory escapes system
}

/**
 * Predict spacecraft trajectory with NO thrust.
 * Used to show "where you're going" ballistically.
 */
export const predictTrajectory = (
  state: SimState,
  duration: number,
  resolution: number = 500
): PredictionResult => {
  const dt = duration / resolution;
  const points: TrajectoryPoint[] = [];
  
  // Clone state with thrust disabled
  let simState: SimState = {
    ...state,
    isThrusting: false,
    rotationInput: 0,
  };
  
  let impactBody: Body | null = null;
  let impactTime: number | null = null;
  
  for (let i = 0; i < resolution; i++) {
    simState = simulationStep(simState, dt);
    
    points.push({
      position: simState.spacecraft.position,
      velocity: simState.spacecraft.velocity,
      time: simState.time,
    });
    
    // Check for collision with any body
    for (const body of simState.bodies) {
      const dist = mag(sub(simState.spacecraft.position, body.position));
      if (dist < body.radius) {
        impactBody = body;
        impactTime = simState.time;
        break;
      }
    }
    
    if (impactBody) break;
  }
  
  // Check if escaping (very simplified - just check if going away fast)
  const lastPoint = points[points.length - 1];
  const escapes = lastPoint && mag(lastPoint.velocity) > 20000; // rough escape velocity
  
  return { points, impactBody, impactTime, escapes };
};

/**
 * Predict trajectory WITH current thrust maintained.
 * Shows "where will I go if I keep thrusting."
 */
export const predictWithThrust = (
  state: SimState,
  thrustDuration: number,
  coastDuration: number,
  resolution: number = 500
): PredictionResult => {
  const totalDuration = thrustDuration + coastDuration;
  const dt = totalDuration / resolution;
  const thrustSteps = Math.floor(thrustDuration / dt);
  
  const points: TrajectoryPoint[] = [];
  let simState = state;
  let impactBody: Body | null = null;
  let impactTime: number | null = null;
  
  for (let i = 0; i < resolution; i++) {
    // Thrust for first portion, then coast
    simState = simulationStep(
      { ...simState, isThrusting: i < thrustSteps && state.isThrusting },
      dt
    );
    
    points.push({
      position: simState.spacecraft.position,
      velocity: simState.spacecraft.velocity,
      time: simState.time,
    });
    
    // Collision check
    for (const body of simState.bodies) {
      const dist = mag(sub(simState.spacecraft.position, body.position));
      if (dist < body.radius) {
        impactBody = body;
        impactTime = simState.time;
        break;
      }
    }
    
    if (impactBody) break;
  }
  
  return { points, impactBody, impactTime, escapes: false };
};
```

---

## HUD Data Computation

```typescript
// =============================================================================
// lib/physics/hud.ts
// Compute derived display data for HUD
// =============================================================================

import type { SimState, HUDData, RelativeState, Body } from './types';
import { computeOrbitalElements, deltaVToCircularize, deltaVHohmann } from './orbits';
import { sub, mag, dot, norm } from './vector';
import { dominantBody } from './gravity';
import { G, MOON } from './constants';

export const computeHUDData = (state: SimState): HUDData => {
  const { spacecraft, bodies } = state;
  
  // Find Earth and Moon
  const earth = bodies.find(b => b.id === 'earth')!;
  const moon = bodies.find(b => b.id === 'moon')!;
  
  // Compute relative states for each body
  const relativeStates: RelativeState[] = bodies
    .filter(b => b.id !== 'sun')  // Don't show Sun in HUD (too far)
    .map(body => {
      const relPos = sub(body.position, spacecraft.position);
      const relVel = sub(body.velocity, spacecraft.velocity);
      const distance = mag(relPos);
      const closingSpeed = -dot(norm(relPos), relVel);  // Positive = approaching
      
      return {
        body,
        distance,
        relativeVelocity: relVel,
        closingSpeed,
        orbitalElements: computeOrbitalElements(spacecraft, body),
      };
    });
  
  // Primary body (strongest gravity)
  const { body: primaryBodyData } = dominantBody(spacecraft.position, bodies);
  const primaryBody = relativeStates.find(rs => rs.body.id === primaryBodyData.id)!;
  
  // Delta-v calculations (relative to Earth for now)
  const earthRelative = relativeStates.find(rs => rs.body.id === 'earth')!;
  const dvCircularize = deltaVToCircularize(spacecraft, earth);
  
  // Delta-v to Moon transfer (simplified: Hohmann from current altitude)
  const currentRadius = earthRelative.distance;
  const moonDistance = mag(sub(moon.position, earth.position));
  const { total: dvMoonTransfer } = deltaVHohmann(
    currentRadius, 
    moonDistance, 
    G * earth.mass
  );
  
  // Time to impact (from trajectory prediction - would need to call predictor)
  // For now, null - will be computed by renderer using prediction
  const timeToImpact = null;
  
  return {
    spacecraftVelocity: mag(spacecraft.velocity),
    spacecraftHeading: (spacecraft.heading * 180 / Math.PI + 360) % 360,
    isThrusting: state.isThrusting,
    relativeStates,
    primaryBody,
    deltaVToCircularize: dvCircularize,
    deltaVToMoonTransfer: dvMoonTransfer,
    timeToImpact,
  };
};
```

---

## Gravity Field Visualization

Two approaches to implement:

### Option A: Color Gradient

```typescript
// Compute gravitational potential at each pixel
// Map to color: deep blue (deep well) → black (flat space)
// Render to offscreen canvas, composite under main view

const renderGravityField = (
  ctx: CanvasRenderingContext2D,
  bodies: readonly Body[],
  viewState: ViewState,
  width: number,
  height: number
) => {
  const imageData = ctx.createImageData(width, height);
  const data = imageData.data;
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const worldPos = screenToWorld({ x, y }, viewState);
      const potential = gravitationalPotential(worldPos, bodies);
      
      // Map potential to color
      // potential is negative (bound) or approaching 0 (far)
      const normalized = Math.min(1, Math.abs(potential) / 1e10);
      const intensity = Math.pow(normalized, 0.3);  // Gamma for visibility
      
      const i = (y * width + x) * 4;
      data[i] = 0;                        // R
      data[i + 1] = intensity * 100;      // G
      data[i + 2] = intensity * 255;      // B
      data[i + 3] = intensity * 200;      // A
    }
  }
  
  ctx.putImageData(imageData, 0, 0);
};
```

### Option B: Grid Distortion

```typescript
// Draw a grid, displace vertices by gravitational potential

const renderGravityGrid = (
  ctx: CanvasRenderingContext2D,
  bodies: readonly Body[],
  viewState: ViewState,
  gridSpacing: number = 50  // pixels
) => {
  ctx.strokeStyle = 'rgba(100, 150, 255, 0.3)';
  ctx.lineWidth = 1;
  
  // Draw horizontal lines with vertical displacement
  for (let y = 0; y < height; y += gridSpacing) {
    ctx.beginPath();
    for (let x = 0; x <= width; x += 5) {
      const worldPos = screenToWorld({ x, y }, viewState);
      const potential = gravitationalPotential(worldPos, bodies);
      const displacement = potential / 1e8;  // Scale for visibility
      
      const screenY = y + displacement;
      if (x === 0) {
        ctx.moveTo(x, screenY);
      } else {
        ctx.lineTo(x, screenY);
      }
    }
    ctx.stroke();
  }
  
  // Similar for vertical lines...
};
```

---

## Input Controls

```
┌─────────────────────────────────────────────────────────────┐
│  KEYBOARD CONTROLS                                          │
│                                                             │
│  ← / A      Rotate counterclockwise                        │
│  → / D      Rotate clockwise                               │
│  ↑ / W      Thrust forward                                 │
│  Space      Pause / Resume                                  │
│                                                             │
│  + / =      Increase time warp                              │
│  - / _      Decrease time warp                              │
│  1-5        Preset time warps (1x, 10x, 100x, 1000x, 10000x)│
│                                                             │
│  Mouse wheel    Zoom in/out                                 │
│  Click + drag   Pan view                                    │
│  F              Focus on spacecraft                         │
│  E              Focus on Earth                              │
│  M              Focus on Moon                               │
│                                                             │
│  G              Toggle gravity field visualization          │
│  T              Toggle trajectory prediction                │
│  O              Toggle orbital paths                        │
│  V              Toggle velocity vectors                     │
│  S              Toggle true scale / compressed              │
└─────────────────────────────────────────────────────────────┘
```

---

## Rendering Coordinate System

```typescript
// World coordinates: meters, heliocentric (Sun at origin)
// Screen coordinates: pixels, (0,0) at top-left

interface CoordinateTransform {
  // World position of screen center
  centerWorld: Vec2;
  // Pixels per meter (zoom level)
  scale: number;
  // Screen dimensions
  width: number;
  height: number;
}

const worldToScreen = (world: Vec2, tx: CoordinateTransform): Vec2 => ({
  x: (world.x - tx.centerWorld.x) * tx.scale + tx.width / 2,
  y: tx.height / 2 - (world.y - tx.centerWorld.y) * tx.scale,  // Y flipped
});

const screenToWorld = (screen: Vec2, tx: CoordinateTransform): Vec2 => ({
  x: (screen.x - tx.width / 2) / tx.scale + tx.centerWorld.x,
  y: (tx.height / 2 - screen.y) / tx.scale + tx.centerWorld.y,
});

// Minimum visual size for objects (in pixels) when zoomed out
const MIN_BODY_RADIUS_PX = 4;
const MIN_SPACECRAFT_SIZE_PX = 8;

const visualRadius = (body: Body, tx: CoordinateTransform): number => {
  const trueRadiusPx = body.radius * tx.scale;
  return Math.max(trueRadiusPx, MIN_BODY_RADIUS_PX);
};
```

---

## Development Phases

### Phase 1: Core Physics + Minimal Rendering (MVP)

- [ ] Project setup (Svelte 5 + TypeScript + Vite)
- [ ] Vector math module with tests
- [ ] Gravity calculations with tests
- [ ] Verlet integrator with tests
- [ ] Orbital elements calculation
- [ ] Basic canvas rendering (Earth + spacecraft only)
- [ ] Keyboard controls (rotate + thrust)
- [ ] Single stable Earth orbit

**Milestone:** Spacecraft orbits Earth, can thrust, physics feels right

### Phase 2: Trajectory Prediction + HUD

- [ ] Trajectory prediction (ballistic)
- [ ] Render predicted trajectory line
- [ ] HUD component (velocity, altitude, orbital params)
- [ ] Delta-v displays
- [ ] Time warp controls
- [ ] Zoom + pan

**Milestone:** User can see where they're going, understand orbital parameters

### Phase 3: Moon + Multi-Body

- [ ] Add Moon with orbital motion
- [ ] Add Sun (even if mostly cosmetic at this scale)
- [ ] Full N-body gravity
- [ ] Relative state displays (distance to Moon, etc.)
- [ ] Hohmann transfer Δv calculations

**Milestone:** Can attempt lunar transfer

### Phase 4: Visualization Polish

- [ ] Gravity field visualization (color gradient)
- [ ] Grid distortion option
- [ ] Orbital path rendering
- [ ] Velocity vectors
- [ ] True scale toggle
- [ ] Trail behind spacecraft

**Milestone:** Beautiful, informative visualization

### Phase 5: Educational Features

- [ ] Tutorial/scenario system
- [ ] "Try to reverse your orbit" challenge
- [ ] "Reach the Moon" challenge
- [ ] Explanatory annotations
- [ ] Side panel with orbital mechanics explanations

**Milestone:** Self-contained teaching tool

---

## Future Enhancements (v2+)

- **Mars and interplanetary transfers** — Add Mars, demonstrate planetary alignments, multi-year missions
- **3D rendering** — Three.js version with orbital inclinations
- **Rocket equation** — Fuel mass, specific impulse, limited Δv budget
- **Patched conics visualization** — Show sphere-of-influence transitions
- **Historical missions** — Recreate Apollo trajectories
- **Multiplayer** — Share scenarios, race to the Moon
- **Mobile/touch controls** — Gesture-based thrust and rotation

---

## References

- Vallado, D. A. *Fundamentals of Astrodynamics and Applications*
- Battin, R. H. *An Introduction to the Mathematics and Methods of Astrodynamics*
- NASA JPL Horizons (for accurate ephemeris data)
- Kerbal Space Program (for UI/UX inspiration)

---

## Key Equations Reference

**Vis-viva equation (relates velocity to position in orbit):**
$$v^2 = \mu \left( \frac{2}{r} - \frac{1}{a} \right)$$

**Orbital period:**
$$T = 2\pi \sqrt{\frac{a^3}{\mu}}$$

**Specific orbital energy:**
$$\epsilon = \frac{v^2}{2} - \frac{\mu}{r} = -\frac{\mu}{2a}$$

**Circular orbital velocity:**
$$v_{circ} = \sqrt{\frac{\mu}{r}}$$

**Escape velocity:**
$$v_{esc} = \sqrt{\frac{2\mu}{r}} = \sqrt{2} \cdot v_{circ}$$

**Hohmann transfer Δv:**
$$\Delta v_1 = \sqrt{\frac{\mu}{r_1}} \left( \sqrt{\frac{2r_2}{r_1 + r_2}} - 1 \right)$$
$$\Delta v_2 = \sqrt{\frac{\mu}{r_2}} \left( 1 - \sqrt{\frac{2r_1}{r_1 + r_2}} \right)$$

---

*Document generated from conversation with Claude. Ready for implementation in Claude Code.*
