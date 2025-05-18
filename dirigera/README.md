# MCP-Agents
Example code using MCPs and Agents for home automation.

## Dirigera
Dirigera is MCP and Agents for home automation via the IKEA dirigera hub.

### Installation
You will need a Dirigera Hub and then use the dirigera library (uv run ikea.py) to get a token. First enter the ip-address of your Dirigera Hub in the config.toml file.

You will need to use uv to run the example code.
https://docs.astral.sh/uv/getting-started/installation/


### Usage
Without token in you config.toml this will happen:

```bash
uv run ikea.py

No token found in config.toml - will get you a token.
When you got the token please enter it into the config.toml file.
Input the ip address of your Dirigera then hit ENTER ...
[...]

```

With token this will happen:

```bash
uv run ikea.py
[{
    "id": "bfe2a5cd-41a2-4061-834d-107a875fdb3a_1", 
    "type": "sensor", 
    "deviceType": "environmentSensor", 
    "createdAt": "2024-09-20T19:37:49.000Z", 
    "isReachable": true, 
    "lastSeen": "2025-05-18T18:46:37.000Z", 
    "attributes": {"customName": "Sensor Luft K\u00f6k", "model": "VINDSTYRKA", "manufacturer": "IKEA of Sweden", "firmwareVersion": "1.0.11", "hardwareVersion": "1", "serialNumber": "881A14FFFE0CD802", "productCode": "E2112", "currentTemperature": 25, "currentRH": 35, "currentPM25": 1, "maxMeasuredPM25": 999, "minMeasuredPM25": 0, "vocIndex": 272, "identifyStarted": "2000-01-01T00:00:00.000Z", "identifyPeriod": 0, "permittingJoin": false, "otaStatus": "upToDate", "otaState": "readyToCheck", "otaProgress": 0, "otaPolicy": "autoUpdate", "otaScheduleStart": "00:00", "otaScheduleEnd": "00:00"}, 
    "capabilities": {"canSend": [], "canReceive": ["customName"]},

[...]
Name: Sonoff Temp Vind
Temperature:17.3
Humidity: 46
VoC: None
PM2.5: None
[...]

```

Lots of data is available from the Dirigera Hub (mostly JSON formatted).
