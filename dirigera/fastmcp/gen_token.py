#!/usr/bin/env python3
"""Verbose Dirigera token generator"""
import sys
from dirigera.hub.auth import send_challenge, get_token, ALPHABET, CODE_LENGTH
import random

def random_code():
    return ''.join(random.choice(ALPHABET) for _ in range(CODE_LENGTH))

def main():
    ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.200"
    print(f"Connecting to Dirigera hub at {ip}...")

    code_verifier = random_code()
    print("Sending auth challenge...")

    try:
        code = send_challenge(ip, code_verifier)
        print(f"Challenge accepted!")
        print("")
        print(">>> PRESS THE ACTION BUTTON ON THE HUB <<<")
        input("Then press ENTER here...")
        print("")
        print("Fetching token...")
        token = get_token(ip, code, code_verifier)
        print("")
        print("SUCCESS! Your new token:")
        print(token)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
