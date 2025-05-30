
from xled_plus import *
from xled.discover import discover
from xled_plus.highcontrol import HighControlInterface
from fastmcp import FastMCP
from fastmcp.prompts import Message
import argparse

mcp = FastMCP("Led Light String")

ctr = False

#@mcp.prompt()
@mcp.resource("url://service_name")
def get_service_name() -> str:
    """Return the name of the provided service"""
    return mcp.name

@mcp.resource("url://service_init")
def init_service() -> bool:
    """Initialize the service. Is called before the first tool call from the client."""
    global ctr
    print("Service initialized")
    dev = discover()
    host = dev.ip_address
    ctr = HighControlInterface(host)
    return True

@mcp.resource("url://service_exit")
def exit_service() -> bool:
    """Clean up after the service. Is called when the current client is shutting down."""
    print("Service closed down")
    return True

#@mcp.resource("url://service_prompt")
@mcp.prompt()
def get_service_prompt(lang: str) -> Message:
    """Return the system message snippet suitable for this service."""
    languages = { "en": "English",
                  "sv": "Swedish",
                  "de": "Deutch",
                  "fr": "French",
                  "es": "Spanish"}
    if not lang in languages:
        lang = 'en'
    reply_language = languages[lang]
    return Message(f"You are a helpful assistant that can control a led light string. You reply in {reply_language}.")

#@mcp.resource("url://service_augmentation")
@mcp.prompt()
def get_service_augmentation(lang: str) -> str:
    """Return extra information on the current state, to insert before the user prompt"""
    if ctr:
        mode = ctr.get_mode()['mode']
        state = "Off" if mode=="off" else "On"
    else:
        state = "Offline"
    return "Current state of the led string: " + state

@mcp.tool()
def default_action() -> str:
    """This function can be called whenever there is no obvious other function to call."""
    return "Successfully did nothing"

@mcp.tool()
def lights_on() -> str:
    """Turn on the light string."""
    print("Call to lights_on")
    if ctr:
        ctr.turn_on()
    return "Successfully turned lights on"

@mcp.tool()
def lights_off() -> str:
    """Turn off the light string."""
    print("Call to lights_off")
    if ctr:
        ctr.turn_off()
    return "Successfully turned lights off"

@mcp.tool()
def lights_set_color(red: int, green: int, blue: int) -> str:
    """Set the color of the lights to RBG coordinates. Arguments are red, green and blue, which are the RGB components each integers between 0 and 255."""
    if ctr:
        ctr.show_pattern(ctr.make_solid_pattern((red, green, blue)))
    #ctr.show_color((red, green, blue))
    print("Call to lights_set_color with ("+str(red)+", "+str(green)+", "+str(blue)+")")
    return "Successfully set light color to ("+str(red)+", "+str(green)+", "+str(blue)+")"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=mcp.name)
    parser.add_argument('--host', default="127.0.0.1", help='Host to bind to')
    parser.add_argument('--port', default=8000, type=int, help='Port to bind to')
    parser.add_argument('--transport', default="sse", help='Transport to use (stdio, sse or http)')
    args = parser.parse_args()
    if args.transport != "stdio":
        mcp.run(transport=args.transport, host=args.host, port=args.port)
    else:
        mcp.run()
