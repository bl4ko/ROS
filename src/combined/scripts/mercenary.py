class MercenaryInfo:
    """
    This class represents the information that the mercenary has about the task.
    """

    def __init__(
        self,
        name: str = None,
        cylinder_color: str = None,
        ring_color: str = None,
        wanted_price: int = None,
    ) -> None:
        self.name: str = name
        self.cylinder_color: str = cylinder_color
        self.ring_color: str = ring_color
        self.wanted_price: int = wanted_price

    def __str__(self) -> str:
        return (
            f"Name: {self.name}, Cylinder color: {self.cylinder_color}, Ring color:"
            f" {self.ring_color}, Wanted price: {self.wanted_price}"
        )

    def is_complete(self) -> bool:
        """
        Returns true if mercenary has enough information to complete the task, false otherwise
        """
        return (
            self.name is not None
            and self.cylinder_color is not None
            and self.ring_color is not None
            and self.wanted_price is not None
        )

    @staticmethod
    def are_complete(mercenary_info_list) -> bool:
        """
        Returns true if all mercenaries in the list have enough
        information to complete the task, false otherwise
        """
        valid = True
        for mercenary_info in mercenary_info_list:
            if not mercenary_info.is_complete():
                valid = False
        return valid
