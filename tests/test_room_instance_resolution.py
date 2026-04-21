from __future__ import annotations

from vlmaps.utils.room_provider import RoomProvider, room_command_matches


class _FakeRoomProvider(RoomProvider):
    def __init__(self, rooms):
        self._rooms = list(rooms)

    def get_room_at_cell(self, row: int, col: int):
        return None

    def get_room_centroid(self, room_name: str):
        return None

    def list_rooms(self):
        return list(self._rooms)

    def is_available(self):
        return True


def _provider():
    return _FakeRoomProvider(
        [
            "entryway",
            "bathroom",
            "bathroom.001",
            "bathroom.002",
            "bedroom",
            "bedroom.001",
            "bedroom.002",
        ]
    )


def test_resolve_room_name_keeps_base_room():
    provider = _provider()
    assert provider.resolve_room_name("bathroom") == "bathroom"
    assert provider.resolve_room_name("bedroom") == "bedroom"


def test_resolve_room_name_supports_numeric_aliases():
    provider = _provider()
    assert provider.resolve_room_name("bathroom 1") == "bathroom.001"
    assert provider.resolve_room_name("bathroom 2") == "bathroom.002"
    assert provider.resolve_room_name("bedroom 1") == "bedroom.001"


def test_resolve_room_name_supports_word_aliases_and_exact_instance():
    provider = _provider()
    assert provider.resolve_room_name("bathroom one") == "bathroom.001"
    assert provider.resolve_room_name("bedroom.002") == "bedroom.002"


def test_find_room_mentions_prefers_exact_instances():
    provider = _provider()
    mentions = provider.find_room_mentions("go to bathroom 1 and then bedroom two")
    assert mentions == ["bathroom.001", "bedroom.002"]


def test_room_command_matches_base_and_exact_instance_correctly():
    assert room_command_matches("bathroom.001", "bathroom")
    assert room_command_matches("bathroom.001", "bathroom 1")
    assert room_command_matches("bathroom.001", "bathroom.001")
    assert not room_command_matches("bathroom", "bathroom.001")
    assert not room_command_matches("bathroom.002", "bathroom 1")
