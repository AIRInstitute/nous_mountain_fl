"""
01_check_elevation.py

Diagnostic script for the `Mountain` map (nous_mountain_carla).

Confirms the elevation caveat documented in README.md §4:
  - CARLA's navigable waypoint network reports z = 0 for this map's
    OpenDRIVE network, even though the .xodr itself carries real
    <elevationProfile> data and the visual mesh has real terrain.
  - The vehicle's own transform pitch is a reliable, non-zero road-grade
    signal that varies along the route.

Usage:
    python config.py -m Mountain --delta-seconds 0.05   # in one terminal
    python 01_check_elevation.py                        # in another
"""

import sys
import time

import carla


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    map_name = world.get_map().name
    if "Mountain" not in map_name:
        print(f"Warning: current map is '{map_name}', expected 'Mountain'. "
              "Load it first with: python config.py -m Mountain")
        sys.exit(1)

    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    if not spawn_points:
        print("No spawn points found on this map.")
        sys.exit(1)

    # --- Waypoint elevation check ---
    sample_waypoints = carla_map.generate_waypoints(distance=50.0)
    z_values = [wp.transform.location.z for wp in sample_waypoints]
    max_abs_z = max(abs(z) for z in z_values) if z_values else 0.0

    print(f"Sampled {len(sample_waypoints)} waypoints along the road network.")
    print(f"Max |z| across waypoints: {max_abs_z:.4f} m")
    if max_abs_z < 1e-3:
        print("-> Confirmed: waypoint elevation is effectively z = 0. "
              "This matches the documented CARLA/OpenDRIVE limitation.")
    else:
        print("-> Unexpected: waypoints report non-zero elevation. "
              "Verify spawn points and waypoint coverage.")

    # --- Vehicle pitch as road-grade signal ---
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    vehicle.set_autopilot(True)

    print("\nSpawned a Tesla Model 3 and enabled autopilot.")
    print("Sampling vehicle pitch (road grade proxy) for 10 seconds...\n")

    pitches = []
    try:
        for _ in range(20):
            world.tick() if world.get_settings().synchronous_mode else time.sleep(0.5)
            pitch = vehicle.get_transform().rotation.pitch
            pitches.append(pitch)
            print(f"  pitch = {pitch:+.2f} deg")
    finally:
        vehicle.destroy()

    if pitches:
        print(f"\nPitch range observed: {min(pitches):+.2f} to {max(pitches):+.2f} deg")
        if max(pitches) - min(pitches) > 0.5:
            print("-> Confirmed: vehicle pitch varies meaningfully along the "
                  "route. Use vehicle.get_transform().rotation.pitch as your "
                  "road-grade signal, not waypoint elevation.")
        else:
            print("-> Pitch variation was small in this short sample; try a "
                  "longer run or a steeper section of the route.")


if __name__ == "__main__":
    main()
