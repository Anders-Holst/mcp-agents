#----------------

from fastmcp import FastMCP
from fastmcp.prompts import Message
import argparse
import random

from camera import *
from robotarm import *
from transtable import *

mcp = FastMCP("Candytron 4000")

use_robot = True
use_camera = True

#@mcp.prompt()
@mcp.resource("url://service_name")
def get_service_name() -> str:
    """Return the name of the provided service"""
    return mcp.name

@mcp.resource("url://service_init")
def init_service() -> bool:
    """Initialize the service. Is called before the first tool call from the client."""
    init_ned(use_robot=use_robot)
    print('Initialized robot arm: Niryo Ned 2')

    if init_cam(use_camera=use_camera):
        print('Initialized camera and YOLO model')
    else:
        print('ERROR: failed to initialize camera and YOLO model')
        return False

    cal = False
    for i in range(0, 10):
        if calibrate_positions(3, 4):
            cal = True
            break
    if not cal:
        print("Failed to calibrate positions from camera - make sure the area is visible and put one candy in each corner, and try again.")
        exit_cam()
        return False
    return True

@mcp.resource("url://service_exit")
def exit_service() -> bool:
    """Clean up after the service. Is called when the current client is shutting down."""
    exit_ned()
    exit_cam()
    print('Exiting ned2 and camera')
    return True


@mcp.prompt()
def get_service_prompt(lang: str) -> Message:
    """Return the system message snippet suitable for this service."""
    names = { "en": "Candy Tron",
              "sv": "Kandutron",
              "de": "Candy Tronn",
              "fr": "Candue Tronne",
              "es": "Candy Tron"}
    languages = { "en": "English",
                  "sv": "Swedish",
                  "de": "Deutch",
                  "fr": "French",
                  "es": "Spanish"}
    if not lang in languages:
        lang = 'en'
    name = names[lang]
    reply_language = languages[lang]
    return Message(f"Your name is {name}. You are situated at an exhibition to demonstrate how several AI systems can be connected, such as speech recognition, a large language model, speech synthesis, computer vision, and a robot arm. You are this system. Specifically, you have a robot arm, which allows you to move different types of candy between different positions on a table. You can chat with the visitors, and they may ask about your demonstration. They may also ask you to move candy around on the table or to give them some specific candy. When you know what specific candy on the table the user wants (but not before), you hand it out to them by moving it to the special position O0. Information on the latest positions of candy and their characteristics will be regularly provided by the vision system, for you to internally look up information needed to answer questions or perform moves. However, you never give this type of lists directly to the user. Your replies are friendly, concise and as plain text with no formatting. Your replies are in {reply_language}")

def scene_message(scene, lang):
    if not lang in transhead:
        lang = 'en'
    content = transhead[lang]
    if scene:
        for k in scene:
            if scene[k] in transtable:
                obj,cha = transtable[scene[k]][lang]
                content += "\n" + k + " : " + obj + ", " + cha + "."
    else:
        content += "No candies observed"
    return content

@mcp.prompt()
def get_service_augmentation(lang: str) -> Message:
    """Return extra information on the current state, to insert before the user prompt"""
    scene = acquire_scene()
    return Message(scene_message(scene, lang))

@mcp.tool()
def show_demo_move() -> str:
    scenepos = list(acquire_scene().keys())
    emptypos = [k for k in camera_positions() if k not in scenepos]
    if len(scenepos) and len(emptypos):
        p1 = random.choice(scenepos)
        p2 = random.choice(emptypos)
        if ned_move_between(p1, p2):
            return "Successfully demonstrated a move with the robot arm from " + p1 + " to " + p2
        else:
            return "Failed to move"
    elif not len(emptypos):
        return "No empty positions"
    else:
        return "No candies observed"

@mcp.tool()
def move_between(src: str, dst: str) -> str:
    """Move an object from one position to another position, using the robot arm. The argument 'src' is the current position of the object. The argument 'dst' is the destination position of the object."""
    if ned_move_between(src, dst):
        return f"Successfully moved from {src} to {dst}"
    else:
        return "Failed to move"

@mcp.tool()
def default_action() -> str:
    """This function can be called whenever there is no obvious other function to call."""
    return "Successfully did nothing"

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
