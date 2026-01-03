import numpy as np

class CBBADispatcher:
    """
    Consensus-Based Bundle Algorithm (CBBA) for Task Allocation.
    """
    def __init__(self, num_stations=2):
        self.num_stations = num_stations
        # Simple Kalman Filter State (Just basic smoothing for now)
        self.filtered_states = {} 
    
    def predict_state_kalman(self, agent_id, noisy_pos, dt=1.0):
        """
        Simple 1D Kalman-like filter for position smoothing.
        In real prod, this would be a full 6DOF EKF.
        """
        if agent_id not in self.filtered_states:
            self.filtered_states[agent_id] = noisy_pos
        
        # Prediction (Constant velocity model assumption) + Update
        # Simple exponential smoothing for demo efficiency
        alpha = 0.7
        predicted = self.filtered_states[agent_id] * (1-alpha) + noisy_pos * alpha
        self.filtered_states[agent_id] = predicted
        return predicted

    def assign_tasks(self, swarm_packets):
        """
        Main CBBA Logic.
        1. Receive Bids (Battery Urgency).
        2. Resolve Conflicts.
        3. Output Assignments.
        """
        assignments = {}
        station_queues = {i: [] for i in range(self.num_stations)}
        
        # 1. Collect Bids
        bids = []
        for agent_id, data in swarm_packets.items():
            battery = data['battery']
            pos = np.array(data['position'])
            
            # Smooth position
            clean_pos = self.predict_state_kalman(agent_id, pos)
            
            if battery < 20.0: # Critical Threshold
                # Create a bid: (Priority, AgentID)
                # Priority = 100 - Battery + Distance penalty
                bids.append({
                    'agent': agent_id,
                    'bid_value': (100 - battery),
                    'type': 'CHARGE'
                })
        
        # 2. Consensus / Winner Determination
        # Sort by highest bid
        bids.sort(key=lambda x: x['bid_value'], reverse=True)
        
        # Assign to limited slots (e.g., 1 slot per station for demo)
        slots_available = [True] * self.num_stations
        
        for bid in bids:
            # Find open station
            assigned = False
            for s_id in range(self.num_stations):
                if slots_available[s_id]:
                    # Assign
                    assignments[bid['agent']] = {
                        'type': 'GO_TO_STATION',
                        'station_id': s_id
                    }
                    slots_available[s_id] = False # Slot taken
                    assigned = True
                    break
            
            if not assigned:
                # No slots left, enter queue or conserve mode
                assignments[bid['agent']] = {'type': 'WAIT'}
                
        return assignments
