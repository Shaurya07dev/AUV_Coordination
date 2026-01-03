import numpy as np

class RobotPhysics:
    """
    Handles 3D kinematics and physics for AUVs.
    """
    def __init__(self, delta_t=1.0):
        self.dt = delta_t
        self.drag_coeff = 0.1
        self.max_speed = 2.0  # m/s
        self.battery_capacity = 100.0 # units
        
        # Power consumption rates
        self.base_drain = 0.01  # Drain per second just for being on
        self.move_drain_factor = 0.05 # Drain per unit of speed

    def update_position(self, position, velocity, bounds):
        """
        Updates position based on velocity, applies bounds.
        position: np.array [x, y, z]
        velocity: np.array [vx, vy, vz]
        bounds: np.array [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
        """
        # Simple Euler integration
        new_pos = position + velocity * self.dt
        
        # Clamp to world bounds
        for i in range(3):
            new_pos[i] = np.clip(new_pos[i], bounds[i][0], bounds[i][1])
            
        return new_pos

    def calculate_battery_drain(self, velocity):
        """
        Calculates battery drain based on speed.
        """
        speed = np.linalg.norm(velocity)
        drain = self.base_drain + (speed * self.move_drain_factor) * self.dt
        return drain

    def check_collision(self, pos1, pos2, radius=1.0):
        """
        Simple spherical collision check.
        """
        dist = np.linalg.norm(pos1 - pos2)
        return dist < (radius * 2)

    def apply_ocean_current(self, velocity, current_vector):
        """
        Adds ocean current drift to the AUV's velocity.
        """
        # In a real physics model, current affects force/acceleration.
        # For this kinematic model, we simply add the current vector to velocity.
        return velocity + current_vector
