import time
import warnings

import paho.mqtt.client as mqtt

from utils.GlobalVarGetter import GlobalVarGetter


def connect_to_mqtt(client, host="broker.emqx.io", port=1883):
    if host == "broker.emqx.io":
        warnings.warn("You are using the public MQTT broker, please consider using your own broker.")
    client.connect(host, port, 60)
    client.loop_start()


class MQTTClientSingleton:
    client = None
    uid = None

    @staticmethod
    def init():
        def on_connect(client, userdata, flags, rc, properties):
            print("Connected to mqtt server with result code " + str(rc))

        global_var = GlobalVarGetter.get()
        config = global_var["config"]["global"]["mqtt"]
        MQTTClientSingleton.uid = global_var["config"]["global"]["uid"]
        MQTTClientSingleton.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        MQTTClientSingleton.client.on_connect = on_connect
        host = config["host"] if "host" in config else "broker.emqx.io"
        port = config["port"] if "port" in config else 1883
        connect_to_mqtt(MQTTClientSingleton.client, host, port)

    @staticmethod
    def get_client():
        if MQTTClientSingleton.client is None:
            MQTTClientSingleton.init()
        return MQTTClientSingleton.client

    @staticmethod
    def get_uid():
        if MQTTClientSingleton.uid is None:
            MQTTClientSingleton.init()
        return MQTTClientSingleton.uid
