"""
Module containing the dialogue classes.
"""
import rospy
from sound import SoundPlayer


class PersonDialogue:
    """
    Class represeting dialogue with a person.
    """

    def __init__(self):
        self.name: str = None
        self.cylinder_color: str = None
        self.sound_player = SoundPlayer()

    def start(self):
        """
        Start the dialogue with the user.
        """
        self.sound_player.say("Do you know the wanted person")
        self.name = input("Do you know the wanted person [Yes/No]: ").lower()

        if self.name == "yes":
            self.sound_player.say("Tell me the name of the wanted person")
            self.name = input("Tell me the name of the wanted person: ").lower()

            self.sound_player.say("Tell me the color of cylinder")
            self.cylinder_color = input(
                "Tell me the color of cylinder [Red, Blue, Green, Yellow]:"
            ).lower()

    def is_valid(self):
        """
        Check if the dialogue is valid.
        """
        return self.name is not None and self.cylinder_color is not None


class PosterDialogue:
    """
    Class representing dialogue with a poster.
    """

    def __init__(self, detected_text: str):
        self.detected_text: str = detected_text
        self.name: str = None
        self.wanted_price: str = None
        self.ring_color: str = None
        self.sound_player = SoundPlayer()

    def start(self):
        "Start the poster dialogue"
        rospy.loginfo(f"I detected the following text on the poster: {self.detected_text}")
        self.sound_player.say("Who is in this poster")
        self.name = input("Who is in this poster? ").lower()

        self.sound_player.say("How much is the reward")
        self.wanted_price = int(input("How much is the reward? ").lower())

        self.sound_player.say("What color is the prison ring")
        self.ring_color = input("What color is the prison ring? ").lower()

    def is_valid(self):
        """
        Check if the dialogue is valid.
        """
        return (
            self.name is not None and self.wanted_price is not None and self.ring_color is not None
        )
