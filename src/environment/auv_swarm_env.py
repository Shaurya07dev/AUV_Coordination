from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
from src.environment.robot_physics import RobotPhysics

class AUVSwarmEnv(ParallelEnv):
    metadata = {'render_modes': ['human'], 'name': 'auv_swarm_v0'}

    def __init__(self, render_mode=None):
        self.possible_agents = [f"auv_{i}" for i in range(5)]
        self.agents = self.possible_agents[:]
        self.render_mode = render_mode
        
        # World Config (100m x 100m x 50m deep)
        self.bounds = np.array([[0, 100], [0, 100], [0, 50]])
        self.physics = RobotPhysics(delta_t=0.5) # 0.5s per step
        
        # Docking Stations (Static for now)
        # Station 0: Charging | Station 1: Maintenance
        self.stations = {
            0: np.array([10.0, 10.0, 45.0]), 
            1: np.array([90.0, 90.0, 45.0])
        }

        # Action Space: Continuous [vx, vy, vz]
        self.action_spaces = {agent: spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32) for agent in self.agents}
        
        # Observation Space: 
        # [x, y, z, vx, vy, vz, battery, status_flag] (Size 8)
        self.observation_spaces = {agent: spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32) for agent in self.agents}

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.positions = {a: np.random.rand(3) * [100, 100, 50] for a in self.agents}
        self.velocities = {a: np.zeros(3) for a in self.agents}
        self.batteries = {a: 100.0 for a in self.agents}
        self.states = {a: 1.0 for a in self.agents} # 1.0 = OK, 0.0 = Fault
        
        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions):
        rewards = {}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}

        for agent in self.agents:
            action = actions[agent] # [vx, vy, vz]
            
            # 1. Physics Update
            # Get ocean current (Placeholder for Veros integration)
            current = self._get_ocean_current(self.positions[agent])
            
            # Apply physics
            effective_velocity = self.physics.apply_ocean_current(action, current)
            self.positions[agent] = self.physics.update_position(
                self.positions[agent], 
                effective_velocity, 
                self.bounds
            )
            
            # 2. Battery Update
            drain = self.physics.calculate_battery_drain(effective_velocity)
            self.batteries[agent] -= drain
            
            # Check for death/stuck
            if self.batteries[agent] <= 0:
                self.batteries[agent] = 0
                terminations[agent] = True # Agent dies
                rewards[agent] = -100.0
            else:
                rewards[agent] = -0.1 # Slight time penalty to encourage efficiency
                
            # Charging Logic
            if self._is_at_station(agent, 0): # Charging station
                self.batteries[agent] = min(100.0, self.batteries[agent] + 5.0)
                rewards[agent] += 1.0 # Reward for charging

        observations = {a: self._get_obs(a) for a in self.agents}
        
        # Filter out dead agents
        self.agents = [a for a in self.agents if not terminations[a]]
        
        return observations, rewards, terminations, truncations, infos

    def _get_obs(self, agent):
        return np.concatenate([
            self.positions[agent],
            self.velocities.get(agent, np.zeros(3)), # Use previous velocity or action
            [self.batteries[agent]],
            [self.states[agent]]
        ], dtype=np.float32)

    def _get_ocean_current(self, position):
        # TODO: Load Veros vector field here. 
        # For now, simplistic drift towards East
        return np.array([0.1, 0.0, 0.0]) 

    def _is_at_station(self, agent, station_id):
        # Check dist < 2.0m
        dist = np.linalg.norm(self.positions[agent] - self.stations[station_id])
        return dist < 2.0
