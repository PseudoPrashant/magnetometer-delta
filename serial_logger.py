import serial
import csv
import datetime
import sys

PORT = 'COM7'
BAUD_RATE = 115200

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"log_{timestamp}.csv"

try:
    ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
    print(f"Listening on {PORT} at {BAUD_RATE} baud...")
    print(f"Writing to {filename}. Press Ctrl+C to stop.")

    # newline='' prevents blank rows on Windows
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        while True:
            if ser.in_waiting > 0:
                try:
                    # Read, decode, and strip trailing whitespace/newlines
                    raw_data = ser.readline().decode('utf-8').strip()
                    
                    if not raw_data:
                        continue
                        
                    # Split by comma if your device sends CSV strings. 
                    # If it sends single values, this just puts it in column A.
                    data_list = raw_data.split(',')
                    writer.writerow(data_list)
                    
                    # Force write to disk immediately so you don't lose data on a crash
                    file.flush() 
                    
                    # Print to console (comment this out if the data rate is too high)
                    print(f"Logged: {data_list}") 

                except UnicodeDecodeError:
                    print("Decode Error: Received garbage. Your device isn't sending clean UTF-8.")

except serial.SerialException as e:
    print(f"PORT ERROR: Failed to open {PORT}.")
    print("Is the device unplugged? Is it currently open in the Arduino IDE Serial Monitor or PuTTY? Close them first.")
    print(f"Details: {e}")
    sys.exit(1)
except KeyboardInterrupt:
    print("\nProcess terminated by user.")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Port closed cleanly.")