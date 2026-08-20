from pythonosc.udp_client import SimpleUDPClient
import time

client = SimpleUDPClient("127.0.0.1", 7000)   # UE listens here

while True:
    client.send_message("/obj", "pnk 320 210")
    print("sent")
    time.sleep(0.5)