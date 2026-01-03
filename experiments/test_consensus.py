from src.edge.dispatcher import CBBADispatcher
import unittest

class TestDispatcher(unittest.TestCase):
    def test_battery_assignment(self):
        print("Testing CBBA Dispatcher...")
        dispatcher = CBBADispatcher(num_stations=2)
        
        # Mock State: Agent 1 is low battery
        swarm_state = {
            'auv_0': {'battery': 80.0, 'position': [10, 10, 10]},
            'auv_1': {'battery': 15.0, 'position': [50, 50, 50]}, # Needs charge
            'auv_2': {'battery': 90.0, 'position': [20, 20, 20]}
        }
        
        assignments = dispatcher.assign_tasks(swarm_state)
        
        print(f"Assignments: {assignments}")
        
        # Assertions
        self.assertIn('auv_1', assignments)
        self.assertEqual(assignments['auv_1']['type'], 'GO_TO_STATION')
        
        # Agent 0 should not be assigned (or assigned something else/nothing)
        if 'auv_0' in assignments:
            self.assertNotEqual(assignments['auv_0']['type'], 'GO_TO_STATION')

if __name__ == '__main__':
    unittest.main()
