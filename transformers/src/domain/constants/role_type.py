from enum import Enum

class RoleType(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"