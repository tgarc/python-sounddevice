#!/usr/bin/env python3
"""Test priming output buffer."""
import sounddevice as sd

callback_status = sd.CallbackFlags()


def callback(indata, outdata, frames, time, status):
    outdata.fill(0)
    if status.priming_output:
        print("in shape:", indata.shape)
        outdata[0] = 1
    else:
        raise sd.CallbackStop

with sd.Stream(channels=2, callback=callback,
               prime_output_buffers_using_stream_callback=True):
    print("#" * 80)
    print("press Return to quit")
    print("#" * 80)
    input()
