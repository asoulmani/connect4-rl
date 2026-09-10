from pydantic import BaseModel


class AgentInfo(BaseModel):
    id: str
    name: str
    subtitle: str
    available: bool
