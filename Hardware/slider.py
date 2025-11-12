import tkinter as tk
import serial
import time
import serial.tools.list_ports
from tkinter import messagebox

def find_arduino():
    """Auto-detect Arduino COM port"""
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if ("Arduino" in p.description) or ("CH340" in p.description) or ("USB Serial" in p.description):
            return p.device
    return None

# Try to find Arduino
arduino_port = find_arduino()
if not arduino_port:
    messagebox.showerror("Error", "No Arduino detected. Please connect your board and restart.")
    exit()

try:
    ser = serial.Serial(arduino_port, 9600, timeout=1)
    time.sleep(2)  # wait for Arduino to reset
except Exception as e:
    messagebox.showerror("Connection Error", f"Could not open {arduino_port}:\n{e}")
    exit()

# GUI setup
root = tk.Tk()
root.title("Arduino Servo Controller")

last_sent = 0

def move_servo(value):
    global last_sent
    now = time.time()
    try:
        if now - last_sent > 0.1:  # limit writes to every 100ms
            ser.write((value + "\n").encode())
            last_sent = now
    except serial.SerialException:
        messagebox.showerror("Connection Lost", "Lost connection to Arduino. Please reconnect.")
        root.destroy()

slider = tk.Scale(
    root,
    from_=0,
    to=180,
    orient='horizontal',
    length=400,
    label="Servo Angle",
    command=move_servo
)
slider.set(90)
slider.pack(padx=20, pady=20)

root.mainloop()