import copy
import matplotlib.pyplot as plt
import numpy as np

from opendbc.can.parser import CANParser
from opendbc.car.subaru.values import CanBus, DBC

from openpilot.tools.lib.logreader import LogReader
from selfdrive.pandad.pandad_api_impl import can_capnp_to_list

# An example of searching through a database of segments for a specific condition, and plotting the results.

segments = [
    "c3d1ccb52f5f9d65|2023-07-22--01-23-20/6:10",
]
platform = "SUBARU_OUTBACK"

if __name__ == "__main":
    """
    In this example, we search for positive transitions of Steer_Warning, which indicate that the EPS
    has stopped responding to our messages. This analysis would allow you to find the cause of these
    steer warnings and potentially work around them.
    """
    from openpilot.tools.lib.logreader import LogReader

    lr = LogReader(path)
    print("B")

    for segment in segments:
        lr = LogReader(segment)

        can_msgs = [msg for msg in lr if msg.which() == "can"]

        messages = [
            ("Steering_Torque", 50)
        ]

        cp = CANParser(DBC[platform]["pt"], messages, CanBus.main)

        steering_torque_history = []
        examples = []

        print("A")

        for msg in can_msgs:
            cp.update_strings(can_capnp_to_list([msg.as_builder().to_bytes()]))
            steering_torque_history.append(copy.copy(cp.vl["Steering_Torque"]))

        steer_warning_last = False
        for i, steering_torque_msg in enumerate(steering_torque_history):
            steer_warning = steering_torque_msg["Steer_Warning"]

            steer_angle = steering_torque_msg["Steering_Angle"]

            if steer_warning and not steer_warning_last:  # positive transition of "Steer_Warning"
                examples.append(i)

            steer_warning_last = steer_warning

        FRAME_DELTA = 100  # plot this many frames around the positive transition

        for example in examples:
            fig, axs = plt.subplots(2)

            min_frame = int(example - FRAME_DELTA / 2)
            max_frame = int(example + FRAME_DELTA / 2)

            steering_angle_history = [msg["Steering_Angle"] for msg in steering_torque_history[min_frame:max_frame]]
            steering_warning_history = [msg["Steer_Warning"] for msg in steering_torque_history[min_frame:max_frame]]

            xs = np.arange(-FRAME_DELTA / 2, FRAME_DELTA / 2)

            axs[0].plot(xs, steering_angle_history)
            axs[0].set_ylabel("Steering Angle (deg)")
            axs[1].plot(xs, steering_warning_history)
            axs[1].set_ylabel("Steer Warning")

            plt.show()

