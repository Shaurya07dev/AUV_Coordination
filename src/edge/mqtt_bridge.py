import paho.mqtt.client as mqtt
import json
import json
import time

class MQTTBridge:
    def __init__(self, broker_address="test.mosquitto.org", port=1883):
        self.client = mqtt.Client(protocol=mqtt.MQTTv311)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.broker_address = broker_address
        self.port = port
        self.incoming_commands = []

    def connect(self):
        try:
            self.client.connect(self.broker_address, self.port, 60)
            self.client.loop_start()
            print(f"Connected to MQTT Broker: {self.broker_address}")
        except Exception as e:
            print(f"Failed to connect to MQTT: {e}")

    def on_connect(self, client, userdata, flags, rc):
        print(f"MQTT Connected with result code {rc}")
        self.client.subscribe("auv/cloud/commands/#")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            self.incoming_commands.append(payload)
        except Exception as e:
            print(f"Error parsing MQTT message: {e}")

    def publish_state(self, state_data):
        """
        Publishes the full swarm state to the cloud.
        Topic: auv/edge/state
        """
        payload = json.dumps(state_data)
        self.client.publish("auv/edge/state", payload)

    def get_latest_commands(self):
        """
        Returns and clears the buffer of incoming commands.
        """
        cmds = self.incoming_commands[:]
        self.incoming_commands = []
        return cmds

    def close(self):
        self.client.loop_stop()
        self.client.disconnect()
