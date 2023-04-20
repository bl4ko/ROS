"""
Helper module for playing sounds in brain.py.
"""

from sound_play.libsoundplay import SoundClient


class SoundPlayer:
    """
    This class is responsible for playing sounds when the robot is greeting a face.
    """

    def __init__(self):
        self.sound_client = SoundClient()

    def play_greeting_sound(self):
        """
        Plays a greeting sound.
        """
        self.sound_client.say("Hello, nice to meet you!")
