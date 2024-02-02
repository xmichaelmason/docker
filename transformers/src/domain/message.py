from pydantic import BaseModel
from src.domain.constants.role_type import RoleType


class Message(BaseModel):
    role: RoleType
    content: str
