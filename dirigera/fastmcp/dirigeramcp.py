# MCP Server for controlling Dirigera devices (and reading out data from them)
# Supports outlets, environment sensors and lights.
# Author: Joakim Eriksson
# Created: 2025
#
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
import toml, argparse, dirigera, jwt, secrets, datetime, uvicorn

mcp = FastMCP(name="Dirigera Server")
client = None

# JWT configuration (set at startup when --auth is used)
JWT_SECRET = None
JWT_EXPIRY_HOURS = 24
API_KEY = None


@mcp.custom_route("/auth/token", methods=["POST"])
async def get_token(request: Request) -> JSONResponse:
    """Exchange API key for a JWT token."""
    body = await request.json()
    if body.get("api_key") != API_KEY:
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    now = datetime.datetime.now(datetime.timezone.utc)
    token = jwt.encode(
        {"sub": "mcp-client", "iat": now, "exp": now + datetime.timedelta(hours=JWT_EXPIRY_HOURS)},
        JWT_SECRET, algorithm="HS256",
    )
    return JSONResponse({"token": token, "expires_in": JWT_EXPIRY_HOURS * 3600})


@mcp.custom_route("/auth/refresh", methods=["POST"])
async def refresh_token(request: Request) -> JSONResponse:
    """Refresh a valid (or recently expired) JWT token."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return JSONResponse({"error": "Missing Bearer token"}, status_code=401)
    try:
        jwt.decode(auth[7:], JWT_SECRET, algorithms=["HS256"], leeway=datetime.timedelta(hours=1))
    except jwt.InvalidTokenError as e:
        return JSONResponse({"error": f"Invalid token: {e}"}, status_code=401)
    now = datetime.datetime.now(datetime.timezone.utc)
    token = jwt.encode(
        {"sub": "mcp-client", "iat": now, "exp": now + datetime.timedelta(hours=JWT_EXPIRY_HOURS)},
        JWT_SECRET, algorithm="HS256",
    )
    return JSONResponse({"token": token, "expires_in": JWT_EXPIRY_HOURS * 3600})


def device_info(device, **extra):
    return {'id': device.id, 'name': device.attributes.custom_name,
            'is_reachable': device.is_reachable, 'last_seen': str(device.last_seen), **extra}

@mcp.tool()
def get_environment_sensors() -> list:
    """Lists all environment sensors with their current data such as temperature, humidity, CO2, etc."""
    return [device_info(s, temperature=s.attributes.current_temperature,
                        humidity=s.attributes.current_r_h, pm25=s.attributes.current_p_m25,
                        voc=s.attributes.voc_index, co2=getattr(s.attributes, 'current_c_o2', None))
            for s in client.get_environment_sensors()]

@mcp.tool()
def get_outlets() -> list:
    """Lists all outlets with their current data such as power, voltage, current, etc."""
    return [device_info(o, power=o.attributes.current_active_power,
                        voltage=o.attributes.current_voltage, current=o.attributes.current_amps)
            for o in client.get_outlets()]

@mcp.tool()
def get_lights() -> list:
    """Lists all lights with their current data such as brightness, etc. Null values means that the light does not support that feature."""
    return [device_info(l, light_level=l.attributes.light_level,
                        color_temperature=l.attributes.color_temperature,
                        color_saturation=l.attributes.color_saturation,
                        color_hue=l.attributes.color_hue, is_on=l.attributes.is_on)
            for l in client.get_lights()]

@mcp.tool()
def set_onoff(name: str, is_on: bool) -> str:
    """Set outlet or light status of a named outlet or light. Arguments are on/off"""
    try:
        outlet = client.get_outlet_by_name(name)
        outlet.set_on(outlet_on=is_on)
        return f"outlet {name} set to {is_on}."
    except:
        try:
            light = client.get_light_by_name(name)
            light.set_light(lamp_on=is_on)
            return f"light {name} set to {is_on}."
        except:
            return f"Outlet/Light '{name}' not found"

@mcp.tool()
def set_light_level(name: str, light_level: int) -> str:
    """Set light status of a named light. Arguments is light_level(int)"""
    light = client.get_light_by_name(name)
    if light is None:
        return f"Light '{name}' not found"
    light.set_light_level(light_level=light_level)
    return f"light {name} set to level {light_level}."

@mcp.tool()
def set_light_color(name: str, color_saturation: float, color_hue: float) -> str:
    """Set light status of a named light. Arguments are color_saturation (float 0.0-1.0), color_hue (float, 0-360)"""
    light = client.get_light_by_name(name)
    if light is None:
        return f"Light '{name}' not found"
    if "colorHue" in light.capabilities.can_receive:
        light.set_light_color(hue=color_hue, saturation=color_saturation)
    else:
        return f"Light '{name}' does not support color hue and saturation"
    return f"light {name} set to color {color_saturation} and {color_hue}."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dirigera MCP Server for IoT device management')
    parser.add_argument('--config-path', default="../config.toml", help='Path to Dirigera server configuration file (TOML format)')
    parser.add_argument('--host', default="127.0.0.1", help='Host to bind to (use 0.0.0.0 for remote access)')
    parser.add_argument('--port', default=8000, type=int, help='Port to bind to')
    parser.add_argument('--transport', default="stdio", choices=["stdio", "sse", "streamable-http"],
                        help='Transport: stdio (local), sse (remote), streamable-http (remote, recommended)')
    parser.add_argument('--auth', action='store_true', help='Enable JWT authentication')
    args = parser.parse_args()
    conf = toml.load(args.config_path)
    hub_host = conf['dirigera']['host']
    token = conf['dirigera']['token']
    client = dirigera.Hub(token=token, ip_address=hub_host)

    if args.auth and args.transport in ("streamable-http", "sse"):
        from fastmcp.server.auth.providers.jwt import JWTVerifier

        # Load or generate secrets from config
        JWT_SECRET = conf.get('auth', {}).get('jwt_secret', secrets.token_hex(32))
        API_KEY = conf.get('auth', {}).get('api_key', secrets.token_hex(16))
        JWT_EXPIRY_HOURS = conf.get('auth', {}).get('expiry_hours', 24)

        if 'auth' not in conf:
            conf['auth'] = {'jwt_secret': JWT_SECRET, 'api_key': API_KEY, 'expiry_hours': JWT_EXPIRY_HOURS}
            with open(args.config_path, 'w') as f:
                toml.dump(conf, f)
            print(f"Generated auth config saved to {args.config_path}")

        # Use FastMCP's built-in JWT auth
        mcp.auth = JWTVerifier(public_key=JWT_SECRET, algorithm="HS256")

        print(f"JWT authentication enabled")
        print(f"API Key: {API_KEY}")
        print(f"Token endpoint: POST http://{args.host}:{args.port}/auth/token")

    if args.transport in ("streamable-http", "sse"):
        app = mcp.http_app(transport=args.transport)
        app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["mcp-session-id"])
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        mcp.run()
