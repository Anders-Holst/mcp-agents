#
# Program for testing the IKEA Dirigera library.
#
#
import sys
import dirigera
import json
import toml

# Load configuration
conf = toml.load('config.toml')

host = conf['dirigera']['host']

if not 'token' in conf['dirigera']:
    print("No token found in config.toml - will get you a token.")
    print("When you got the token please enter it into the config.toml file.")
    import dirigera.hub.auth
    dirigera.hub.auth.main()
    sys.exit(1)

token = conf['dirigera']['token']

hub = dirigera.Hub(token=token, ip_address=host)

devices = hub.get("/devices")
print(json.dumps(devices))


outlets = hub.get_outlets()
for outlet in outlets:
    print("Name:", outlet.attributes.custom_name)
    print("Power:", outlet.attributes.current_active_power)

lights = hub.get_lights()
for light in lights:
    print("Name:", light.attributes.custom_name)
    print("In on:", light.attributes.is_on)
    print("In on:", light.attributes.light_level)

sensors = hub.get_environment_sensors()
txt = ""
for sensor in sensors:
    print("Id:", sensor.id)
    id = sensor.id
    dev = hub.get("/devices/" + id)
    print(dev)
    
    print("Name:", sensor.attributes.custom_name)
    print(f"Temperature:{sensor.attributes.current_temperature}")
    print("Humidity:", sensor.attributes.current_r_h)
    print("VoC:", sensor.attributes.voc_index)
    print("PM2.5:", sensor.attributes.current_p_m25)
    print(sensor.attributes)

    

outlets = hub.get_outlets()
txt = ""
for outlet in outlets:
    dict = {'id': outlet.id, 'name': outlet.attributes.custom_name,
            'power': outlet.attributes.current_active_power,
            'voltage': outlet.attributes.current_voltage,
            'current': outlet.attributes.current_amps
    }
    txt = txt + json.dumps(dict) + "\n"
print(txt)
