# src/simulation/environment.py
import numpy as np
import pandas as pd


class SimulationEnvironment:
    """
    Manages the real-time state of the entire simulation world.
    This is the core of the dynamic simulation, responsible for advancing time,
    updating the state of all assets (vehicles, stations), and triggering alarms.
    """

    def __init__(self, data, config):
        self.config = config
        self.data = data
        self.time = 0.0  # Current simulation time in hours

        # --- Vehicle States ---
        self.vehicle_states = {
            vid: {
                'soc': v['initial_soc'],
                'location': v['depot_id'],  # Current node location
                'status': 'IDLE',  # IDLE, DRIVING, SWAPPING, UNLOADING
                'action_end_time': 0.0,
                'route_plan': [],
                'tasks_completed': [],
                'total_wait_time': 0.0
            } for vid, v in data['vehicles'].items()
        }

        # --- Station States ---
        self.station_states = {
            sid: {
                'bess_soc': config.BESS_CAPACITY_KWH * 0.5,
                'queue': [],  # List of vehicle IDs waiting
                'availability': 2,  # Number of available service bays
                'grid_power_draw': 0.0
            } for sid in data['stations']
        }

        # --- Grid States ---
        self.grid_bus_loads = {bus_id: 0.0 for bus_id in data['station_to_bus_map'].values()}
        print("Simulation Environment Initialized.")

    def advance_time_step(self):
        """Advances the simulation time by one step and returns any triggered alarms."""
        self.time += self.config.TIME_STEP_HOURS
        alarms = self._check_for_alarms()
        return alarms

    def _check_for_alarms(self):
        """Checks for all alarm conditions (grid, station, vehicle)."""
        alarms = []
        # 1. Station Queue Alarm
        for sid, state in self.station_states.items():
            if len(state['queue']) > self.config.STATION_QUEUE_ALARM_THRESHOLD:
                alarm = {'type': 'STATION_CONGESTION', 'station_id': sid, 'details': f"Queue is {len(state['queue'])}"}
                print(f"  -> ALARM @ T={self.time:.2f}h: {alarm['type']} at {sid}")
                alarms.append(alarm)

        # 2. Grid Overload Alarm (Simulated)
        for bus_id, load in self.grid_bus_loads.items():
            if load > self.config.GRID_BUS_ALARM_MW * 1000:  # Convert MW to kW
                affected_stations = [s for s, b in self.data['station_to_bus_map'].items() if b == bus_id]
                alarm = {'type': 'GRID_OVERLOAD', 'bus_id': bus_id, 'affected_stations': affected_stations}
                print(f"  -> ALARM @ T={self.time:.2f}h: {alarm['type']} at Bus {bus_id}")
                alarms.append(alarm)

        # 3. Vehicle Traffic Jam Alarm (Simulated)
        for vid, state in self.vehicle_states.items():
            if state['status'] == 'DRIVING' and np.random.rand() < self.config.TRAFFIC_JAM_PROBABILITY:
                alarm = {'type': 'TRAFFIC_JAM', 'vehicle_id': vid}
                print(f"  -> ALARM @ T={self.time:.2f}h: {alarm['type']} for vehicle {vid}")
                alarms.append(alarm)

        return alarms

    def get_vehicle_state_for_replan(self, vehicle_id):
        """Gathers the current, precise state of a vehicle needed for re-planning."""
        state = self.vehicle_states[vehicle_id]
        return {
            'current_time': self.time,
            'current_location': state['location'],
            'current_soc': state['soc'],
        }