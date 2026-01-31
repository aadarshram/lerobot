#!/usr/bin/env python

"""
Simple script to test the base motor (shoulder_pan) of the follower arm.

This script allows you to:
1. Connect to the base motor
2. Read its current position
3. Move it to specific positions
4. Test the full range of motion

Usage:
    python test_base_motor.py --port /dev/ttyACM0
"""

import argparse
import logging
import time

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_motor_connection(bus: FeetechMotorsBus):
    """Test basic motor connectivity."""
    logger.info("Testing motor connection...")
    
    # Read current position
    position = bus.read("Present_Position", "shoulder_pan")
    logger.info(f"✓ Motor connected successfully! Current position: {position}")
    
    # Read other motor info
    voltage = bus.read("Present_Voltage", "shoulder_pan")
    temperature = bus.read("Present_Temperature", "shoulder_pan")
    logger.info(f"  Voltage: {voltage}V, Temperature: {temperature}°C")
    
    return position


def test_motor_movement(bus: FeetechMotorsBus, current_pos: float):
    """Test basic motor movement."""
    logger.info("\nTesting motor movement...")
    
    # Enable torque
    bus.enable_torque("shoulder_pan")
    logger.info("✓ Torque enabled")
    
    # Small movement test
    logger.info("Moving motor +90 degrees...")
    target_pos = current_pos - 90
    bus.write("Goal_Position", "shoulder_pan", target_pos)
    time.sleep(2)
    
    new_pos = bus.read("Present_Position", "shoulder_pan")
    logger.info(f"✓ Moved to position: {new_pos}")
    
    # Move back
    logger.info("Moving back to original position...")
    bus.write("Goal_Position", "shoulder_pan", current_pos)
    time.sleep(2)
    
    final_pos = bus.read("Present_Position", "shoulder_pan")
    logger.info(f"✓ Returned to position: {final_pos}")


def test_range_of_motion(bus: FeetechMotorsBus):
    """Test the full range of motion interactively."""
    logger.info("\nTesting range of motion...")
    logger.info("Move the motor manually through its full range.")
    logger.info("Positions will be recorded for 10 seconds...")
    
    bus.disable_torque("shoulder_pan")
    
    positions = []
    start_time = time.time()
    
    while time.time() - start_time < 10:
        pos = bus.read("Present_Position", "shoulder_pan")
        positions.append(pos)
        time.sleep(0.1)
    
    min_pos = min(positions)
    max_pos = max(positions)
    
    logger.info(f"✓ Range of motion: {min_pos} to {max_pos}")
    logger.info(f"  Total range: {max_pos - min_pos} degrees")


def interactive_mode(bus: FeetechMotorsBus):
    """Interactive mode for manual testing."""
    logger.info("\n=== Interactive Mode ===")
    logger.info("Commands:")
    logger.info("  'r' - Read current position")
    logger.info("  'm <value>' - Move to position (e.g., 'm 45')")
    logger.info("  't on/off' - Enable/disable torque")
    logger.info("  'q' - Quit")
    
    while True:
        cmd = input("\nEnter command: ").strip().lower()
        
        if cmd == 'q':
            break
        elif cmd == 'r':
            pos = bus.read("Present_Position", "shoulder_pan")
            logger.info(f"Current position: {pos}")
        elif cmd.startswith('m '):
            try:
                target = float(cmd.split()[1])
                bus.enable_torque("shoulder_pan")
                bus.write("Goal_Position", "shoulder_pan", target)
                logger.info(f"Moving to {target}...")
                time.sleep(1)
                actual = bus.read("Present_Position", "shoulder_pan")
                logger.info(f"Actual position: {actual}")
            except (ValueError, IndexError):
                logger.error("Invalid command. Use: m <position>")
        elif cmd == 't on':
            bus.enable_torque("shoulder_pan")
            logger.info("Torque enabled")
        elif cmd == 't off':
            bus.disable_torque("shoulder_pan")
            logger.info("Torque disabled")
        else:
            logger.info("Unknown command")


def main():
    parser = argparse.ArgumentParser(description="Test follower arm base motor")
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyUSB0",
        help="Serial port for the motor controller"
    )
    parser.add_argument(
        "--motor_id",
        type=int,
        default=1,
        help="Motor ID (default: 1 for shoulder_pan/base motor)"
    )
    parser.add_argument(
        "--skip_movement",
        action="store_true",
        help="Skip automatic movement tests"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter interactive mode after tests"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Follower Arm Base Motor Test Script")
    logger.info("=" * 60)
    logger.info(f"Port: {args.port}")
    logger.info(f"Motor ID: {args.motor_id}")
    
    # Create default calibration (allows testing without full calibration)
    # These are safe default values for the STS3215 motor
    default_calibration = {
        "shoulder_pan": MotorCalibration(
            id=args.motor_id,
            drive_mode=0,
            homing_offset=2048,  # Middle position for 12-bit resolution (0-4095)
            range_min=0,         # Minimum encoder value
            range_max=4095,      # Maximum encoder value (12-bit: 2^12 - 1)
        )
    }
    
    logger.info("Using default calibration values (safe for testing)")
    logger.info("  - Center position: 2048")
    logger.info("  - Range: 0 to 4095 (full 360° rotation)")
    
    # Create motor bus with just the base motor (shoulder_pan)
    bus = FeetechMotorsBus(
        port=args.port,
        motors={
            "shoulder_pan": Motor(args.motor_id, "sts3215", MotorNormMode.DEGREES),
        },
        calibration=default_calibration,
    )
    
    try:
        # Connect to the motor
        logger.info("\nConnecting to motor...")
        bus.connect()
        
        # Configure motor for position control
        with bus.torque_disabled():
            bus.write("Operating_Mode", "shoulder_pan", OperatingMode.POSITION.value)
            # Set PID values for smooth operation
            bus.write("P_Coefficient", "shoulder_pan", 16)
            bus.write("I_Coefficient", "shoulder_pan", 0)
            bus.write("D_Coefficient", "shoulder_pan", 32)
        
        logger.info("✓ Motor configured successfully")
        
        # Test connection and read position
        current_pos = test_motor_connection(bus)
        
        # Test movement (if not skipped)
        if not args.skip_movement:
            response = input("\nDo you want to test motor movement? (y/n): ")
            if response.lower() == 'y':
                test_motor_movement(bus, current_pos)
        
        # Test range of motion
        response = input("\nDo you want to test range of motion? (y/n): ")
        if response.lower() == 'y':
            test_range_of_motion(bus)
        
        # Interactive mode
        if args.interactive:
            interactive_mode(bus)
        
        logger.info("\n✓ All tests completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"\n✗ Error: {e}", exc_info=True)
    finally:
        # Disable torque and disconnect
        logger.info("\nCleaning up...")
        try:
            bus.disable_torque()
            bus.disconnect()
            logger.info("✓ Motor disconnected safely")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


if __name__ == "__main__":
    main()
