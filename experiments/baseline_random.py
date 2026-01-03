from src.environment.auv_swarm_env import AUVSwarmEnv
import numpy as np

def main():
    print("Initializing AUV Swarm Environment (3D)...")
    env = AUVSwarmEnv(render_mode='human')
    
    observations, infos = env.reset()
    print(f"Environment created. Agents: {env.agents}")
    
    for episode in range(3):
        print(f"\n--- Episode {episode+1} ---")
        observations, infos = env.reset()
        
        for step in range(20): # Short test
            # Random actions: [vx, vy, vz]
            actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
            
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Print state of first agent
            if "auv_0" in observations:
                pos = observations["auv_0"][0:3]
                batt = observations["auv_0"][6]
                print(f"Step {step}: AUV_0 Pos={pos.round(1)}, Batt={batt:.1f}%")
            
            if not env.agents:
                print("All agents dead/done.")
                break

    print("\nTest Complete! Physics and Env API are working.")

if __name__ == "__main__":
    main()
