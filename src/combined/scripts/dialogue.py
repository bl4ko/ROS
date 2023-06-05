"""
Module containing the dialogue classes.
"""
import rospy
from sound import SoundPlayer

VALID_NAMES = ["john", "joe", "amber"]
VALID_COLORS = ["red", "green", "blue", "yellow"]


def handle_input_type(input_str: str, expected_type: type) -> bool:
    """
    Validates the type of the user input.

    Parameters:
    input: str: User input.
    expected_type: str: The expected type of the input. Can be 'int' or 'str'.

    Returns:
    bool: True if the input matches the expected type, False otherwise.
    """
    if expected_type == int:
        try:
            int(input_str)
            if int(input_str) < 0:
                return False
            return True
        except ValueError:
            return False
    elif expected_type == str:
        return isinstance(input_str, str)
    else:
        raise ValueError(f"Unknown type {expected_type}, expected 'int' or 'str'")


def validate_color(color: str) -> bool:
    """
    Validates the color.

    Parameters:
    color: str: The color to validate.

    Returns:
    bool: True if the color is valid, False otherwise.
    """
    return color in VALID_COLORS


def validate_name(name: str) -> bool:
    """
    Validates the name.

    Parameters:
    name: str: The name to validate.

    Returns:
    bool: True if the name is valid, False otherwise.
    """
    valid_names = ["john", "joe", "amber"]
    return name in valid_names


class PersonDialogue:
    """
    Class representing dialogue with a person.
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
        answer = None

        while True:
            answer = input("Do you know the wanted person [Yes/No]: ").lower()
            if handle_input_type(answer, str):
                break
            self.sound_player.say("Invalid answer. Please answer with yes or no.")

        if answer == "yes":
            self.sound_player.say("Tell me the name of the wanted person")
            while True:
                answer = input("Tell me the name of the wanted person: ").lower()
                if handle_input_type(answer, str) and validate_name(answer):
                    break
                self.sound_player.say(
                    f"Invalid name. Please enter a valid name: {', '.join(VALID_NAMES)}."
                )

            self.sound_player.say("Tell me the color of cylinder")
            while True:
                self.cylinder_color = input(
                    "Tell me the color of cylinder [Red, Blue, Green, Yellow]:"
                ).lower()
                if handle_input_type(self.cylinder_color, str) and validate_color(
                    self.cylinder_color
                ):
                    break

                self.sound_player.say(
                    f"Invalid input. Please enter a valid color: {', '.join(VALID_COLORS)}."
                )

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
        """
        Start the poster dialogue
        """
        rospy.loginfo(f"I detected the following text on the poster: {self.detected_text}")

        self.sound_player.say("Who is in this poster")
        while True:
            self.name = input("Who is in this poster? ").lower()
            if handle_input_type(self.name, str) and validate_name(self.name):
                break
            self.sound_player.say(
                "Invalid name. Please enter a valid name. Valid names are"
                f" {', '.join(VALID_NAMES)}."
            )

        self.sound_player.say("How much is the reward")
        while True:
            self.wanted_price = input("How much is the reward? ").lower()
            if handle_input_type(self.wanted_price, int):
                break
            self.sound_player.say("Invalid input. Please enter a valid number.")

        self.sound_player.say("What color is the prison ring")
        while True:
            self.ring_color = input("What color is the prison ring? ").lower()
            if handle_input_type(self.ring_color, str) and validate_color(self.ring_color):
                break
            self.sound_player.say(
                f"Invalid input. Please enter a valid color: {', '.join(VALID_COLORS)}."
            )

    def is_valid(self):
        """
        Check if the dialogue is valid.
        """
        return (
            self.name is not None and self.wanted_price is not None and self.ring_color is not None
        )
